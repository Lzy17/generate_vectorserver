<!-- LINT.IfChange -->
<h1>Create a stateful custom op</h1>
<p>This example shows you how to create TensorFlow custom ops with internal state.
These custom ops use resources to hold their state. You do this by implementing
a stateful C++ data structure instead of using tensors to hold the state. This
example also covers:</p>
<ul>
<li>Using <code>tf.Variable</code>s for state as an alternative to using a resource (which
    is recommended for all cases where storing the state in fixed size tensors
    is reasonable)</li>
<li>Saving and restoring the state of the custom op by using <code>SavedModel</code>s</li>
<li>The <code>IsStateful</code> flag which is true for ops with state and other unrelated
    cases</li>
<li>A set of related custom ops and ops with various signatures (number of
    inputs and outputs)</li>
</ul>
<p>For additional context,
read the
<a href="https://www.tensorflow.org/guide/create_op">OSS guide on creating custom ops</a>.</p>
<h2>Background</h2>
<h3>Overview of resources and ref-counting</h3>
<p>TensorFlow C++ resource classes are derived from the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_base.h"><code>ResourceBase</code> class</a>.
A resource is referenced in the Tensorflow Graph as a
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_handle.h"><code>ResourceHandle</code></a>
Tensor, of the type
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto"><code>DT_RESOURCE</code></a>.
A resource can be owned by a ref-counting <code>ResourceHandle</code>, or by a
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_mgr.h"><code>ResourceMgr</code></a>
(deprecated for new ops).</p>
<p>A handle-owned resource using ref-counting is automatically destroyed when all
resource handles pointing to it go out of scope. If the resource needs to be
looked up from a name, for example, if the resource handle is serialized and
deserialized, the resource must be published as an unowned entry with
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_mgr.h#L641"><code>ResourceMgr::CreateUnowned</code></a>
upon creation. The entry is a weak reference to the resource, and is
automatically removed when the resource goes out of scope.</p>
<p>In contrast, with the deprecated <code>ResourceMgr</code> owned resources, a resource
handle behaves like a weak ref - it does not destroy the underlying resource
when its lifetime ends.
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/resource_variable_ops.cc#L339"><code>DestroyResourceOp</code></a>
must be explicitly called to destroy the resource. An example of why requiring
calling <code>DestroyResourceOp</code> is problematic is that it is easy to introduce a bug
when adding a new code path to an op that returns without calling
<code>DestroyResourceOp</code>.</p>
<p>While it would be possible to have a data structure that has a single operation
implemented by a single custom op (for example, something that just has a 'next
state' operation that requires no initialization), typically a related set of
custom ops are used with a resource. <strong>Separating creation and use into
different custom ops is recommended.</strong> Typically, one custom op creates the
resource and one or more additional custom ops implement functionality to access
or modify it.</p>
<h3>Using <code>tf.Variable</code>s for state</h3>
<p>An alternative to custom ops with internal state is to store the state
externally in one or more
<a href="https://www.tensorflow.org/api_docs/python/tf/Variable"><code>tf.Variable</code></a>s as
tensors (as detailed <a href="https://www.tensorflow.org/guide/variable">here</a>) and have
one or more (normal, stateless) ops that use tensors stored in these
<code>tf.Variable</code>s. One example is <code>tf.random.Generator</code>.</p>
<p>For cases where using variables is possible and efficient, using them is
preferred since the implementation does not require adding a new C++ resource
type. An example of a case where using variables is not possible and custom ops
using a resource must be used is where the amount of space for data grows
dynamically.</p>
<p>Here is a toy example of using the
<code>multiplex_2</code>
custom op with a <code>tf.Variable</code> in the same manner as an in-built TensorFlow op
is used with variables. The variable is initialized to 1 for every value.
Indices from a
<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset"><code>tf.data.Dataset</code></a>
cause the corresponding element of the variable to be doubled.</p>
<!-- test_snippets_in_readme skip -->
<p>```python
import tensorflow as tf
from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op</p>
<p>def variable_and_stateless_op():
  n = 10
  v = tf.Variable(tf.ones(n, dtype=tf.int64), trainable=False)
  dataset = tf.data.Dataset.from_tensor_slices([5, 1, 7, 5])
  for position in dataset:
    print(v.numpy())
    cond = tf.one_hot(
        position, depth=n, on_value=True, off_value=False, dtype=bool)
    v.assign(multiplex_2_op.multiplex(cond, v*2, v))
  print(v.numpy())
```</p>
<p>This outputs:</p>
<!-- test_snippets_in_readme skip -->
<p><code>[1 1 1 1 1 1 1 1 1 1]
[1 1 1 1 1 2 1 1 1 1]
[1 2 1 1 1 2 1 1 1 1]
[1 2 1 1 1 2 1 2 1 1]
[1 2 1 1 1 4 1 2 1 1]</code></p>
<p>It is also possible to pass a Python <code>tf.Variable</code> handle as a
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto"><code>DT_RESOURCE</code></a>
input to a custom op where its C++ kernel uses the handle to access the variable
using the variable's internal C++ API. This is not a common case because the
variable's C++ API provides little extra functionality. It can be appropriate
for cases that only sparsely update the variable.</p>
<h3>The <code>IsStateful</code> flag</h3>
<p>All TensorFlow ops have an
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_mgr.h#L505"><code>IsStateful</code></a>
flag. It is set to <code>True</code> for ops that have internal state, both for ops that
follow the recommended pattern of using a resource and those that have internal
state in some other way. In addition, it is also set to <code>True</code> for ops with side
effects (for example, I/O) and for disabling certain optimizations for an op.</p>
<h3>Save and restore state with <code>SavedModel</code></h3>
<p>A <a href="https://www.tensorflow.org/guide/saved_model"><code>SavedModel</code></a> contains a
complete TensorFlow program, including trained parameters (like <code>tf.Variable</code>s)
and computation. In some cases, you can save and restore the state in a custom
op by using <code>SavedModel</code>. This is optional, but is important for some cases, for
example if a custom op is used during training and the state needs to persist
when training is restarted from a checkpoint. The hash table in this example
supports <code>SavedModel</code> with a custom Python wrapper that implements the
<code>_serialize_to_tensors</code> and <code>_restore_from_tensors</code> methods of
<a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/experimental/TrackableResource"><code>tf.saved_model.experimental.TrackableResource</code></a>
and ops that are used by these methods.</p>
<h2>Creating a hash table with a stateful custom op</h2>
<p>This example demonstrates how to implement custom ops that use ref-counting
resources to track state. This example creates a <code>simple_hash_table</code> which is
similar to
<a href="https://www.tensorflow.org/api_docs/python/tf/lookup/experimental/MutableHashTable"><code>tf.lookup.experimental.MutableHashTable</code></a>.</p>
<p>The hash table implements a set of 4 CRUD (Create/Read/Update/Delete) style
custom ops. Each of these ops has a different signature and they illustrate
cases such as custom ops with more than one output.</p>
<p>The hash table supports <code>SavedModel</code> through the <code>export</code> and <code>import</code> ops to
save/restore state. For an actual hash table use case, it is preferable to use
(or extend) the existing
<a href="https://www.tensorflow.org/api_docs/python/tf/lookup"><code>tf.lookup</code></a> ops. In this
simple example, <code>insert</code>, <code>find</code>, and <code>remove</code> use only a single key-value pair
per call. In contrast, existing <code>tf.lookup</code> ops can use multiple key-value
pairs in a single call.</p>
<p>The table below summarizes the six custom ops implemented in this example.</p>
<p>Operation | Purpose                     | Resource class method | Kernel class                    | Custom op                      | Python class member
--------- | --------------------------- | --------------------- | ------------------------------- | ------------------------------ | -------------------
create    | CRUD and SavedModel: create | Default constructor   | <code>SimpleHashTableCreateOpKernel</code> | Examples&gt;SimpleHashTableCreate | <code>__init__</code>
find      | CRUD: read                  | Find                  | <code>SimpleHashTableFindOpKernel</code>   | Examples&gt;SimpleHashTableFind   | <code>find</code>
insert    | CRUD: update                | Insert                | <code>SimpleHashTableInsertOpKernel</code> | Examples&gt;SimpleHashTableInsert | <code>insert</code>
remove    | CRUD: delete                | Remove                | <code>SimpleHashTableRemoveOpKernel</code> | Examples&gt;SimpleHashTableRemove | <code>remove</code>
import    | SavedModel: restore         | Import                | <code>SimpleHashTableImportOpKernel</code> | Examples&gt;SimpleHashTableImport | <code>do_import</code>
export    | SavedModel: save            | Export                | <code>SimpleHashTableExportOpKernel</code> | Examples&gt;SimpleHashTableExport | <code>export</code></p>
<p>You can use this hash table as:</p>
<!-- test_snippets_in_readme skip -->
<p>```python
hash_table = simple_hash_table_op.SimpleHashTable(tf.int32, float,
                                                  default_value=-999.0)
result1 = hash_table.find(key=1, dynamic_default_value=-999.0)</p>
<h1>-999.0</h1>
<p>hash_table.insert(key=1, value=100.0)
result2 = hash_table.find(key=1, dynamic_default_value=-999.0)</p>
<h1>100.0</h1>
<p>```</p>
<p>The example below contains C++ and Python code snippets to illustrate the code
flow. These snippets are not all complete; some are missing namespace
declarations, imports, and test cases.</p>
<p>This example deviates slightly from the general recipe for creating TensorFlow
custom ops. The most significant differences are noted in each step below.</p>
<h3>Step 0 - Implement the resource class</h3>
<p>Implement a
<code>SimpleHashTableResource</code>
resource class derived from
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_base.h"><code>ResourceBase</code></a>.</p>
<p>```live-snippet
template <class K, class V>
class SimpleHashTableResource : public ::tensorflow::ResourceBase {
 public:
  Status Insert(const Tensor&amp; key, const Tensor&amp; value) {
    const K key_val = key.flat<K>()(0);
    const V value_val = value.flat<V>()(0);</p>
<pre><code>mutex_lock l(mu_);
table_[key_val] = value_val;
return OkStatus();
</code></pre>
<p>}</p>
<p>Status Find(const Tensor&amp; key, Tensor* value, const Tensor&amp; default_value) {
    // Note that tf_shared_lock could be used instead of mutex_lock
    // in ops that do not not modify data protected by a mutex, but
    // go/totw/197 recommends using exclusive lock instead of a shared
    // lock when the lock is not going to be held for a significant amount
    // of time.
    mutex_lock l(mu_);</p>
<pre><code>const V default_val = default_value.flat&lt;V&gt;()(0);
const K key_val = key.flat&lt;K&gt;()(0);
auto value_val = value-&gt;flat&lt;V&gt;();
value_val(0) = gtl::FindWithDefault(table_, key_val, default_val);
return OkStatus();
</code></pre>
<p>}</p>
<p>Status Remove(const Tensor&amp; key) {
    mutex_lock l(mu_);</p>
<pre><code>const K key_val = key.flat&lt;K&gt;()(0);
if (table_.erase(key_val) != 1) {
  return errors::NotFound("Key for remove not found: ", key_val);
}
return OkStatus();
</code></pre>
<p>}</p>
<p>// Save all key, value pairs to tensor outputs to support SavedModel
  Status Export(OpKernelContext<em> ctx) {
    mutex_lock l(mu_);
    int64_t size = table_.size();
    Tensor</em> keys;
    Tensor* values;
    TF_RETURN_IF_ERROR(
        ctx-&gt;allocate_output("keys", TensorShape({size}), &amp;keys));
    TF_RETURN_IF_ERROR(
        ctx-&gt;allocate_output("values", TensorShape({size}), &amp;values));
    auto keys_data = keys-&gt;flat<K>();
    auto values_data = values-&gt;flat<V>();
    int64_t i = 0;
    for (auto it = table_.begin(); it != table_.end(); ++it, ++i) {
      keys_data(i) = it-&gt;first;
      values_data(i) = it-&gt;second;
    }
    return OkStatus();
  }</p>
<p>// Load all key, value pairs from tensor inputs to support SavedModel
  Status Import(const Tensor&amp; keys, const Tensor&amp; values) {
    const auto key_values = keys.flat<K>();
    const auto value_values = values.flat<V>();</p>
<pre><code>mutex_lock l(mu_);
table_.clear();
for (int64_t i = 0; i &lt; key_values.size(); ++i) {
  gtl::InsertOrUpdate(&amp;table_, key_values(i), value_values(i));
}
return OkStatus();
</code></pre>
<p>}</p>
<p>// Create a debug string with the content of the map if this is small,
  // or some example data if this is large, handling both the cases where the
  // hash table has many entries and where the entries are long strings.
  std::string DebugString() const override { return DebugString(3); }
  std::string DebugString(int num_pairs) const {
    std::string rval = "SimpleHashTable {";
    size_t count = 0;
    const size_t max_kv_str_len = 100;
    mutex_lock l(mu_);
    for (const auto&amp; pair : table_) {
      if (count &gt;= num_pairs) {
        strings::StrAppend(&amp;rval, "...");
        break;
      }
      std::string kv_str = strings::StrCat(pair.first, ": ", pair.second);
      strings::StrAppend(&amp;rval, kv_str.substr(0, max_kv_str_len));
      if (kv_str.length() &gt; max_kv_str_len) strings::StrAppend(&amp;rval, " ...");
      strings::StrAppend(&amp;rval, ", ");
      count += 1;
    }
    strings::StrAppend(&amp;rval, "}");
    return rval;
  }</p>
<p>private:
  mutable mutex mu_;
  absl::flat_hash_map<K, V> table_ TF_GUARDED_BY(mu_);
};
```</p>
<p>Note that this class provides:</p>
<ul>
<li>Helper methods to access the hash table. These methods correspond to the
    <code>find</code>, <code>insert</code>, and <code>remove</code> ops.</li>
<li>Helper methods to <code>import</code>/<code>export</code> the complete internal state of the hash
    table. These methods help support <code>SavedModel</code>.</li>
<li>A <a href="https://en.wikipedia.org/wiki/Lock_(computer_science)">mutex</a> for the
    helper methods to use for exclusive access to the
    <a href="https://abseil.io/docs/cpp/guides/container#abslflat_hash_map-and-abslflat_hash_set"><code>absl::flat_hash_map</code></a>.
    This ensures thread safety by ensuring that only one thread at a time can
    access the data in the hash table.</li>
</ul>
<h3>Step 1 - Define the op interface</h3>
<p>Define op interfaces and register all the custom ops you create for the hash
table. You typically define one custom op to create the resource. You also
define one or more custom ops that correspond to operations on the data
structure. You also define custom ops to perform import and export operations
that input/output the whole internal state. These ops are optional; you define
them in this example to support <code>SavedModel</code>. As the resource is automatically
deleted based on ref-counting, there is no custom op required to delete the
resource.</p>
<p>The <code>simple_hash_table</code> has kernels for the <code>create</code>, <code>insert</code>, <code>find</code>,
<code>remove</code>, <code>import</code>, and <code>export</code> ops which use the resource object that actually
stores and manipulates data. The resource is passed between ops using a resource
handle. The <code>create</code> op has an <code>Output</code> of type <code>resource</code>. The other ops have
an <code>Input</code> of type <code>resource</code>. The interface definitions for all the ops along
with their shape functions are in
<code>simple_hash_table_op.cc</code>.</p>
<p>Note that these definitions do not explicitly use <code>SetIsStateful</code>. The
<code>IsStateful</code> flag is set automatically for any op with an input or output of
type <code>DT_RESOURCE</code>.</p>
<p>The definitions below and their corresponding kernels and generated Python
wrappers illustrate the following cases:</p>
<ul>
<li>0, 1, 2, and 3 inputs</li>
<li>0, 1, and 2 outputs</li>
<li><code>Attr</code> for types where none are used by <code>Input</code> or <code>Output</code> and all have to
    be explicitly passed into the generated wrapper (for example,
    <code>SimpleHashTableCreate</code>)</li>
<li><code>Attr</code> for types where all are used by <code>Input</code> and/or <code>Output</code> (so they are
    set implicitly inside the generated wrapper) and none are passed into the
    generated wrapper (for example, <code>SimpleHashTableFind</code>)</li>
<li><code>Attr</code> for types where some but not all are used by <code>Input</code> or <code>Output</code> and
    only those that are not used are explicitly passed into the generated
    wrapper (for example, <code>SimpleHashTableRemove</code> where there is an <code>Input</code> that
    uses <code>key_dtype</code> but the <code>value_type</code> <code>Attr</code> is a parameter to the generated
    wrapper).</li>
</ul>
<p><code>REGISTER_OP("Examples&gt;SimpleHashTableCreate")
    .Output("output: resource")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ScalarOutput);</code>
<code>REGISTER_OP("Examples&gt;SimpleHashTableFind")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Input("default_value: value_dtype")
    .Output("value: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ThreeScalarInputsScalarOutput);</code>
<code>REGISTER_OP("Examples&gt;SimpleHashTableInsert")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Input("value: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ThreeScalarInputs);</code>
<code>REGISTER_OP("Examples&gt;SimpleHashTableRemove")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(TwoScalarInputs);</code>
<code>REGISTER_OP("Examples&gt;SimpleHashTableExport")
    .Input("table_handle: resource")
    .Output("keys: key_dtype")
    .Output("values: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ExportShapeFunction);</code>
<code>REGISTER_OP("Examples&gt;SimpleHashTableImport")
    .Input("table_handle: resource")
    .Input("keys: key_dtype")
    .Input("values: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ImportShapeFunction);</code></p>
<h3>Step 2 - Register the op implementation (kernel)</h3>
<p>Declare kernels for specific types of key-value pairs. Register the kernel by
calling the <code>REGISTER_KERNEL_BUILDER</code> macro.</p>
<p><code>#define REGISTER_KERNEL(key_dtype, value_dtype)               \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples&gt;SimpleHashTableCreate")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint&lt;key_dtype&gt;("key_dtype")             \
          .TypeConstraint&lt;value_dtype&gt;("value_dtype"),        \
      SimpleHashTableCreateOpKernel&lt;key_dtype, value_dtype&gt;); \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples&gt;SimpleHashTableFind")                    \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint&lt;key_dtype&gt;("key_dtype")             \
          .TypeConstraint&lt;value_dtype&gt;("value_dtype"),        \
      SimpleHashTableFindOpKernel&lt;key_dtype, value_dtype&gt;);   \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples&gt;SimpleHashTableInsert")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint&lt;key_dtype&gt;("key_dtype")             \
          .TypeConstraint&lt;value_dtype&gt;("value_dtype"),        \
      SimpleHashTableInsertOpKernel&lt;key_dtype, value_dtype&gt;)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples&gt;SimpleHashTableRemove")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint&lt;key_dtype&gt;("key_dtype")             \
          .TypeConstraint&lt;value_dtype&gt;("value_dtype"),        \
      SimpleHashTableRemoveOpKernel&lt;key_dtype, value_dtype&gt;)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples&gt;SimpleHashTableExport")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint&lt;key_dtype&gt;("key_dtype")             \
          .TypeConstraint&lt;value_dtype&gt;("value_dtype"),        \
      SimpleHashTableExportOpKernel&lt;key_dtype, value_dtype&gt;)  \
  REGISTER_KERNEL_BUILDER(                                    \
      Name("Examples&gt;SimpleHashTableImport")                  \
          .Device(DEVICE_CPU)                                 \
          .TypeConstraint&lt;key_dtype&gt;("key_dtype")             \
          .TypeConstraint&lt;value_dtype&gt;("value_dtype"),        \
      SimpleHashTableImportOpKernel&lt;key_dtype, value_dtype&gt;);</code>
<code>REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int32, tstring);
REGISTER_KERNEL(int64_t, double);
REGISTER_KERNEL(int64_t, float);
REGISTER_KERNEL(int64_t, int32);
REGISTER_KERNEL(int64_t, int64_t);
REGISTER_KERNEL(int64_t, tstring);
REGISTER_KERNEL(tstring, bool);
REGISTER_KERNEL(tstring, double);
REGISTER_KERNEL(tstring, float);
REGISTER_KERNEL(tstring, int32);
REGISTER_KERNEL(tstring, int64_t);
REGISTER_KERNEL(tstring, tstring);</code></p>
<h3>Step 3 - Implement the op kernel(s)</h3>
<p>The implementation of kernels for the ops in the hash table all use helper
functions from the <code>SimpleHashTableResource</code> resource class.</p>
<p>You implement the op kernels in two phases:</p>
<ol>
<li>
<p>Implement the kernel for the <code>create</code> op that has a <code>Compute</code> method that
    creates a resource object and a ref-counted handle for it.</p>
<p><!-- test_snippets_in_readme skip -->
<code>c++
handle_tensor.scalar&lt;ResourceHandle&gt;()() =
    ResourceHandle::MakeRefCountingHandle(
        new SimpleHashTableResource&lt;K, V&gt;(), /* … */);</code></p>
</li>
<li>
<p>Implement the kernels(s) for each of the other operations on the resource
    that have Compute methods that get a resource object and use one or more of
    its helper methods.</p>
<p><!-- test_snippets_in_readme skip -->
<code>c++
MyResource* resource;
OP_REQUIRES_OK(ctx, GetResource(ctx, &amp;resource));
// The GetResource local function uses handle.GetResource&lt;resource_type&gt;()
OP_REQUIRES_OK(ctx, resource-&gt;Find(key, out, default_value));</code></p>
</li>
</ol>
<h4>Creating the resource</h4>
<p>The <code>create</code> op creates a resource object of type <code>SimpleHashTableResource</code> and
then uses <code>MakeRefCountingHandle</code> to pass the ownership to a resource handle.
This op outputs a <code>resource</code> handle.</p>
<p>```
template <class K, class V>
class SimpleHashTableCreateOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableCreateOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}</p>
<p>void Compute(OpKernelContext<em> ctx) override {
    Tensor handle_tensor;
    AllocatorAttributes attr;
    OP_REQUIRES_OK(ctx, ctx-&gt;allocate_temp(DT_RESOURCE, TensorShape({}),
                                           &amp;handle_tensor, attr));
    handle_tensor.scalar<ResourceHandle>()() =
        ResourceHandle::MakeRefCountingHandle(
            new SimpleHashTableResource<K, V>(), ctx-&gt;device()-&gt;name(),
            /</em>dtypes_and_shapes=*/{}, ctx-&gt;stack_trace());
    ctx-&gt;set_output(0, handle_tensor);
  }</p>
<p>private:
  // Just to be safe, avoid accidentally copying the kernel.
  TF_DISALLOW_COPY_AND_ASSIGN(SimpleHashTableCreateOpKernel);
};
```</p>
<h4>Getting the resource</h4>
<p>In
<code>simple_hash_table_kernel.cc</code>,
the <code>GetResource</code> helper function uses an input <code>resource</code> handle to retrieve
the corresponding <code>SimpleHashTableResource</code> object. It is used by all the custom
ops that use the resource (that is, all the custom ops in the set other than
<code>create</code>).</p>
<p><code>template &lt;class K, class V&gt;
Status GetResource(OpKernelContext* ctx,
                   SimpleHashTableResource&lt;K, V&gt;** resource) {
  const Tensor&amp; handle_tensor = ctx-&gt;input(0);
  const ResourceHandle&amp; handle = handle_tensor.scalar&lt;ResourceHandle&gt;()();
  typedef SimpleHashTableResource&lt;K, V&gt; resource_type;
  TF_ASSIGN_OR_RETURN(*resource, handle.GetResource&lt;resource_type&gt;());
  return OkStatus();
}</code></p>
<h4>Using the resource</h4>
<p>The ops that use the resource use <code>GetResource</code> to get a pointer to the resource
object and call the corresponding helper function for that object. Below is the
source code for the <code>find</code> op which uses <code>resource-&gt;Find</code>. The other ops that
use the resource similarly use the corresponding helper method in the resource
class.</p>
<p>```
template <class K, class V>
class SimpleHashTableFindOpKernel : public OpKernel {
 public:
  explicit SimpleHashTableFindOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}</p>
<p>void Compute(OpKernelContext<em> ctx) override {
    DataTypeVector expected_inputs = {DT_RESOURCE, DataTypeToEnum<K>::v(),
                                      DataTypeToEnum<V>::v()};
    DataTypeVector expected_outputs = {DataTypeToEnum<V>::v()};
    OP_REQUIRES_OK(ctx, ctx-&gt;MatchSignature(expected_inputs, expected_outputs));
    SimpleHashTableResource<K, V></em> resource;
    OP_REQUIRES_OK(ctx, GetResource(ctx, &amp;resource));
    // Note that ctx-&gt;input(0) is the Resource handle
    const Tensor&amp; key = ctx-&gt;input(1);
    const Tensor&amp; default_value = ctx-&gt;input(2);
    TensorShape output_shape = default_value.shape();
    Tensor* out;
    OP_REQUIRES_OK(ctx, ctx-&gt;allocate_output("value", output_shape, &amp;out));
    OP_REQUIRES_OK(ctx, resource-&gt;Find(key, out, default_value));
  }
};
```</p>
<h4>Compile the op</h4>
<p>Compile the C++ op to create a kernel library and Python wrapper that enables
you to use the op with TensorFlow.</p>
<p>Create a <code>BUILD</code> file for the op which declares the dependencies and the output
build targets. Refer to
<a href="https://www.tensorflow.org/guide/create_op#build_the_op_library">building for OSS</a>.</p>
<p>Note
that you will be reusing this <code>BUILD</code> file later on in this example.</p>
<p>```
tf_custom_op_library(
    name = "simple_hash_table_kernel.so",
    srcs = [
        "simple_hash_table_kernel.cc",
        "simple_hash_table_op.cc",
    ],
    deps = [
        "//third_party/absl/container:flat_hash_map",
        "//third_party/tensorflow/core/lib/gtl:map_util",
        "//third_party/tensorflow/core/platform:strcat",
    ],
)</p>
<p>py_strict_library(
    name = "simple_hash_table_op",
    srcs = ["simple_hash_table_op.py"],
    data = ["simple_hash_table_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)</p>
<p>py_strict_library(
    name = "simple_hash_table",
    srcs = ["simple_hash_table.py"],
    srcs_version = "PY3",
    deps = [
        ":simple_hash_table_op",
        "//third_party/py/tensorflow",
    ],
)
```</p>
<h3>Step 4 - Create the Python wrapper</h3>
<p>To create the Python wrapper, import and implement a function that serves as the
op's public API and provides a docstring.</p>
<p>The Python wrapper for this example, <code>simple_hash_table.py</code> uses the
<code>SimpleHashTable</code> class to provide methods that allow access to the data
structure. The custom ops are used to implement these methods. This class also
supports <code>SavedModel</code> which allows the state of the resource to be saved and
restored for checkpointing. The class is derived from the
<code>tf.saved_model.experimental.TrackableResource</code> base class and implements the
<code>_serialize_to_tensors</code> (which uses the <code>export</code> op) and <code>_restore_from_tensors</code>
(which uses the <code>import</code> op) methods defined by this base class. The <code>__init__</code>
method of this class calls the <code>_create_resource</code> method (which is an override
of a base class method required for <code>SavedModel</code>) which in turn calls the C++
<code>examples_simple_hash_table_create</code> kernel. The resource handle returned by this
kernel is stored in the private <code>self._resource_handle</code> object member. Methods
that use this handle access it using the public <code>self.resource_handle</code> property
provided by the <code>tf.saved_model.experimental.TrackableResource</code> base class.</p>
<p>```
  def _create_resource(self):
    """Create the resource tensor handle.</p>
<pre><code>`_create_resource` is an override of a method in base class
`TrackableResource` that is required for SavedModel support. It can be
called by the `resource_handle` property defined by `TrackableResource`.

Returns:
  A tensor handle to the lookup table.
"""
assert self._default_value.get_shape().ndims == 0
table_ref = gen_simple_hash_table_op.examples_simple_hash_table_create(
    key_dtype=self._key_dtype,
    value_dtype=self._value_dtype,
    name=self._name)
return table_ref
</code></pre>
<p>def _serialize_to_tensors(self):
    """Implements checkpointing protocols for <code>Trackable</code>."""
    tensors = self.export()
    return {"table-keys": tensors[0], "table-values": tensors[1]}</p>
<p>def _restore_from_tensors(self, restored_tensors):
    """Implements checkpointing protocols for <code>Trackable</code>."""
    return gen_simple_hash_table_op.examples_simple_hash_table_import(
        self.resource_handle, restored_tensors["table-keys"],
        restored_tensors["table-values"])
```</p>
<p>The <code>find</code> method in this class calls the <code>examples_simple_hash_table_find</code>
custom op using the reference handle (from the public <code>self.resource_handle</code>
property) and a key and default value. The other methods are similar. It is
recommended that methods for using a resource avoid logic other than a call to a
generated Python wrapper to avoid eager/<code>tf.function</code> inconsistencies (avoid
Python logic that is lost when a <code>tf.function</code> is created). In the case of
<code>find</code>, using <code>tf.convert_to_tensor</code> cannot be avoided and is not lost during
<code>tf.function</code> creation.</p>
<p><code>``
  def find(self, key, dynamic_default_value=None, name=None):
    """Looks up</code>key` in a table, outputs the corresponding value.</p>
<pre><code>The `default_value` is used if key not present in the table.

Args:
  key: Key to look up. Must match the table's key_dtype.
  dynamic_default_value: The value to use if the key is missing in the
    table. If None (by default), the `table.default_value` will be used.
  name: A name for the operation (optional).

Returns:
  A tensor containing the value in the same shape as `key` using the
    table's value type.

Raises:
  TypeError: when `key` do not match the table data types.
"""
with tf.name_scope(name or "%s_lookup_table_find" % self._name):
  key = tf.convert_to_tensor(key, dtype=self._key_dtype, name="key")
  if dynamic_default_value is not None:
    dynamic_default_value = tf.convert_to_tensor(
        dynamic_default_value,
        dtype=self._value_dtype,
        name="default_value")
  value = gen_simple_hash_table_op.examples_simple_hash_table_find(
      self.resource_handle, key, dynamic_default_value
      if dynamic_default_value is not None else self._default_value)
return value
</code></pre>
<p>```</p>
<p>The Python wrapper specifies that gradients are not implemented in this example.
For an example of a differentiable map. For general information about gradients,
read
<a href="https://www.tensorflow.org/guide/create_op#implement_the_gradient_in_python">Implement the gradient in Python</a>
in the OSS guide.</p>
<p><code>tf.no_gradient("Examples&gt;SimpleHashTableCreate")
tf.no_gradient("Examples&gt;SimpleHashTableFind")
tf.no_gradient("Examples&gt;SimpleHashTableInsert")
tf.no_gradient("Examples&gt;SimpleHashTableRemove")</code></p>
<p>The full source code for the Python wrapper is in
<code>simple_hash_table_op.py</code>]
and
<code>simple_hash_table.py</code>.</p>
<h3>Step 5 - Test the op</h3>
<p>Create op tests using classes derived from
<a href="https://www.tensorflow.org/api_docs/python/tf/test/TestCase"><code>tf.test.TestCase</code></a>
and the
<a href="https://github.com/abseil/abseil-py/blob/main/absl/testing/parameterized.py">parameterized tests provided by Abseil</a>.</p>
<p>Create tests using three or more custom ops to create <code>SimpleHashTable</code> objects
and:</p>
<ol>
<li>Update the state using the <code>insert</code>, <code>remove</code>, and/or <code>import</code> methods</li>
<li>Observe the state using the <code>find</code> and/or <code>export</code> methods</li>
</ol>
<p>Here is an example test:</p>
<p><code>def test_find_insert_find_strings_eager(self):
    default = 'Default'
    foo = 'Foo'
    bar = 'Bar'
    hash_table = simple_hash_table.SimpleHashTable(tf.string, tf.string,
                                                   default)
    result1 = hash_table.find(foo, default)
    self.assertEqual(result1, default)
    hash_table.insert(foo, bar)
    result2 = hash_table.find(foo, default)
    self.assertEqual(result2, bar)</code></p>
<p>Create a helper function to work with the hash table:</p>
<p><code>def _use_table(self, key_dtype, value_dtype):
    hash_table = simple_hash_table.SimpleHashTable(key_dtype, value_dtype, 111)
    result1 = hash_table.find(1, -999)
    hash_table.insert(1, 100)
    result2 = hash_table.find(1, -999)
    hash_table.remove(1)
    result3 = hash_table.find(1, -999)
    results = tf.stack((result1, result2, result3))
    return results  # expect [-999, 100, -999]</code></p>
<p>The following test explicitly creates a <code>tf.function</code> with <code>_use_table</code>.
Ref-counting causes the C++ resource object created in <code>_use_table</code> to be
destroyed when this <code>tf.function</code> returns inside <code>self.evaluate(results)</code>. By
explicitly creating the <code>tf.function</code> instead of relying on decorators like
<code>@test_util.with_eager_op_as_function</code> and
<code>@test_util.run_in_graph_and_eager_modes</code> (such as in
<code>multiplex_1</code>),
you have an explicit place in the test corresponding to where the C++ resource's
destructor is called.</p>
<p>This test also shows a test that is parameterized for different data types; it
is actually two tests, one with <code>tf.int32</code> input / <code>float</code> output and the other
with <code>tf.int64</code> input / <code>tf.int32</code> output.</p>
<p><code>def test_find_insert_find_tf_function(self, key_dtype, value_dtype):
    results = def_function.function(
        lambda: self._use_table(key_dtype, value_dtype))
    self.assertAllClose(self.evaluate(results), [-999.0, 100.0, -999.0])</code></p>
<p>Reuse the <code>BUILD</code> file to add build
rules for the Python API wrapper and the op test.</p>
<p>```
tf_custom_op_library(
    name = "simple_hash_table_kernel.so",
    srcs = [
        "simple_hash_table_kernel.cc",
        "simple_hash_table_op.cc",
    ],
    deps = [
        "//third_party/absl/container:flat_hash_map",
        "//third_party/tensorflow/core/lib/gtl:map_util",
        "//third_party/tensorflow/core/platform:strcat",
    ],
)</p>
<p>py_strict_library(
    name = "simple_hash_table_op",
    srcs = ["simple_hash_table_op.py"],
    data = ["simple_hash_table_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)</p>
<p>py_strict_library(
    name = "simple_hash_table",
    srcs = ["simple_hash_table.py"],
    srcs_version = "PY3",
    deps = [
        ":simple_hash_table_op",
        "//third_party/py/tensorflow",
    ],
)</p>
<p>tf_py_test(
    name = "simple_hash_table_test",
    size = "medium",  # This test blocks because it writes and reads a file,
    timeout = "short",  # but it still runs quickly.
    srcs = ["simple_hash_table_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
        "no_pip",
    ],
    deps = [
        ":simple_hash_table",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python/framework:errors",
        "//third_party/tensorflow/python/framework:test_lib",
    ],
)
```</p>
<p>Test the op by running:</p>
<!-- test_snippets_in_readme skip -->
<p><code>shell
$ bazel test //third_party/tensorflow/google/g3doc/example/simple_hash_table:simple_hash_table_test</code></p>
<h3>Use the op</h3>
<p>Use the op by importing and calling it as follows:</p>
<!-- test_snippets_in_readme skip -->
<p>```python
import tensorflow as tf</p>
<p>from tensorflow.examples.custom_ops_doc.simple_hash_table import simple_hash_table</p>
<p>hash_table = simple_hash_table.SimpleHashTable(tf.int32, float, -999.0)
result1 = hash_table.find(1, -999.0)  # -999.0
hash_table.insert(1, 100.0)
result2 = hash_table.find(1, -999.0)  # 100.0
```</p>
<p>Here, <code>simple_hash_table</code> is the name of the Python wrapper that was created
in this example.</p>
<h3>Summary</h3>
<p>In this example, you learned how to implement a simple hash table data structure
using stateful custom ops.</p>
<p>The table below summarizes the build rules and targets for building and testing
the <code>simple_hash_table</code> op.</p>
<p>Op components                           | Build rule             | Build target               | Source
--------------------------------------- | ---------------------- | -------------------------- | ------
Kernels (C++)                           | <code>tf_custom_op_library</code> | <code>simple_hash_table_kernel</code> | <code>simple_hash_table_kernel.cc</code>, <code>simple_hash_table_op.cc</code>
Wrapper (automatically generated)       | N/A.                   | <code>gen_simple_hash_table_op</code> | N/A
Wrapper (with public API and docstring) | <code>py_strict_library</code>    | <code>simple_hash_table_op</code>, <code>simple_hash_table</code>     | <code>simple_hash_table_op.py</code>, <code>simple_hash_table.py</code>
Tests                                   | <code>tf_py_test</code>           | <code>simple_hash_table_test</code>   | <code>simple_hash_table_test.py</code></p>
<!-- LINT.ThenChange(simple_hash_table.md) -->