<!-- LINT.IfChange -->
<h1>Create a custom multiplexer op with C++ backward compatibility</h1>
<p>This guide provides an end-to-end implementation of a new custom op that is
backwards compatible with an existing custom op.</p>
<p>The example in this guide implements a new custom op that handles inputs that
are Python lists of tensors, and is backwards compatible with an existing op
that only handles inputs that are single tensors.</p>
<p>The existing op is a multiplexer that returns elements chosen from either of two
single input tensors <code>x</code> or <code>y</code> depending on a single <code>condition</code> tensor.</p>
<p>The content on this page assumes familiarity with the high-level process for
adding custom ops to TensorFlow. For additional context, read the
<a href="https://www.tensorflow.org/guide/create_op">OSS guide on creating custom ops</a>.</p>
<h2>A backwards compatible kernel that handles lists of tensors</h2>
<p>This example demonstrates how you can create a custom multiplexer,
<code>multiplex_4</code>, to register a new kernel that is backward compatible with an
existing multiplex_2` op.</p>
<p>The new custom op registers a kernel
(multiplex_4_kernel.cc) that takes lists of tensors as inputs, and is backwards
compatible with the existing kernel (multiplex_2_kernel.cc) that takes only
single tensors as inputs.</p>
<p>The <code>multiplex_4</code> op is similar to
<a href="https://numpy.org/doc/stable/reference/generated/numpy.select.html">numpy.select</a>,
while the <code>multiplex_2</code> op is similar to
<a href="https://numpy.org/doc/stable/reference/generated/numpy.where.html">numpy.where</a>.</p>
<p>The lists of tensors that the new op takes as inputs are of a particular fixed
size. Since the list size is defined in <code>Attr</code>, it is fixed at graph
construction time when the constructor of the C++ kernel is called. Therefore,
the size of the list cannot be data dependent. See
<a href="https://www.tensorflow.org/guide/ragged_tensor">Ragged tensors</a> for
variable length lists.</p>
<p>This example contains C++ and Python code snippets to illustrate the code flow.
These snippets may be missing namespace declarations, imports, and test cases.</p>
<h3>Prerequsites - Implement <code>multiplex_2</code> and <code>SavedModel</code></h3>
<p>This example uses a <a href="https://www.tensorflow.org/guide/saved_model"><code>SavedModel</code></a>
from an existing <code>multiplex_2</code> custom op.</p>
<p>The <code>muliplex_2_save.py</code> file uses <code>save</code> from <code>model_using_muliplex.py</code> to
create a <code>SavedModel</code> named <code>model_using_multiplex</code> in the current working
directory.</p>
<p><code>``
def save(multiplex_op, path):
  """Save a model that contains the given</code>multiplex_op`.</p>
<p>Args:
    multiplex_op: A multiplex Custom Op, e.g. multiplex_4_op.multiplex. This is
      parameterized so it can also be used to create an "old" model with an
      older version of the op, e.g. multiplex_2_op.multiplex.
    path: Directory to save model to.
  """
  example_cond, example_a, example_b = _get_example_tensors()</p>
<p>class UseMultiplex(tf.Module):</p>
<pre><code>@tf.function(input_signature=[
    tf.TensorSpec.from_tensor(example_cond),
    tf.TensorSpec.from_tensor(example_a),
    tf.TensorSpec.from_tensor(example_b)
])
def use_multiplex(self, cond, a, b):
  return multiplex_op(cond, a, b)
</code></pre>
<p>model = UseMultiplex()
  tf.saved_model.save(
      model,
      path,
      signatures=model.use_multiplex.get_concrete_function(
          tf.TensorSpec.from_tensor(example_cond),
          tf.TensorSpec.from_tensor(example_a),
          tf.TensorSpec.from_tensor(example_b)))
```</p>
<p>This <code>SavedModel</code> has the old version of the custom op (<code>multiplex_2</code>) that only
supports individual tensors as inputs. The following steps will register a
kernel that accepts lists of tensors as inputs, while maintaining backward
compatability with the previous op.</p>
<h3>Step 1 - Define the op interface</h3>
<p>Define the op interface and register it using the <code>REGISTER_OP</code> macro.</p>
<p><code>``
REGISTER_OP("Examples&gt;MultiplexDense")
    .Input("cond: N * bool")
    .Input("a_values: N * T")
    .Input("b_values: T")
    .Output("output_values: T")
    .Attr("T: type")
    .Attr("N: int = 1")
    .SetShapeFn(MultiplexShapeFunction)
    .Doc(R"doc(
Return elements chosen from</code>a_values<code>or</code>b_values<code>depending on</code>cond`.</p>
<p>When <code>a_values</code> and <code>cond</code> are tenors (i.e. N=1), this is similar to <code>np.where</code>
and <code>tf.where</code>. When <code>a_values</code> and <code>cond</code> are lists of tensors (i.e. N&gt;1),
this is similar to <code>np.select</code>. In either case these are simplified to only
handle dense tensors, no optional parameters, no broadcasting, etc..</p>
<p>cond: tf.Tensor or list of tf.Tensor of type bool. If it is a list, <code>a_values</code>
      must be a list of the same length. Where True, yield the corresponding
      element from <code>a_values</code> (with priority to the first one encountered in
      lists), otherwise yield <code>b_values</code>.
a_values: tf.Tensor or list of tf.Tensor. Each tensor has the same type and
          shape as <code>b_values</code>. If it is a list, <code>cond</code> must be a list of the
          same length.
b_values: tf.Tensor with the same type and shape as the <code>a_values</code> if it is a
          tensor or as every element of <code>a_values</code> if <code>a_values</code> is a list.
output_values: A tf.Tensor with elements from <code>a_values</code> where <code>cond</code> is True,
               and elements from <code>b</code> elsewhere.
)doc");
```</p>
<p>While the <code>multiplex_2</code> op defined inputs as single tensors, such as <code>cond:
bool</code> and <code>a_values: T</code>, this op supports lists of tensors by adding <code>N*</code>, where
<code>N</code> is the length of the lists.</p>
<p>The default list size (<code>N</code>) is set to 1 with the following: <code>.Attr("N: int =
1")</code>. If the inputs are single tensors, then <code>N</code> is equal to 1, which is
backwards compatible with a previous definition of <code>.Input("x: T")</code>.</p>
<p>All lists in this example are of equal length (<code>N</code>). To support lists of
different lengths, define an attribute for each unique length. For example:</p>
<!-- test_snippets_in_readme skip -->
<p><code>c++
.Input("short_list: short_len * float")
.Input("long_list: long_len * float")
.Attr("short_len: int = 1")
.Attr("long_len: int &gt;= 10")</code></p>
<h3>Step 2 - Register the op implementation (kernel)</h3>
<p>The C++ kernel in <code>multiplex_4_kernel.cc</code> implements a multiplexer that accepts
lists of tensors as inputs. Register the kernel by calling the
<code>REGISTER_KERNEL_BUILDER</code> macro.</p>
<p>```
#define REGISTER_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Examples&gt;MultiplexDense")       \
                              .Device(::tensorflow::DEVICE_CPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexDenseOp<type>)</p>
<p>TF_CALL_ALL_TYPES(REGISTER_KERNELS);</p>
<h1>undef REGISTER_KERNELS</h1>
<p>```</p>
<h3>Step 3 - Implement the op kernel</h3>
<p>In the <code>multiplex_4_kernel.cc</code> op kernel, create a class derived from
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#:~:text=class%20OpKernel"><code>OpKernel</code></a>
that implements a <code>Compute</code> method. This method retrieves and validates input
tensors, performs computation, and creates output tensors.</p>
<p>```
template <typename T>
class MultiplexDenseOp : public OpKernel {
 public:
  explicit MultiplexDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx-&gt;GetAttr("N", &amp;num_cond_a_));
  }</p>
<p>MultiplexDenseOp(const MultiplexDenseOp&amp; other) = delete;
  MultiplexDenseOp&amp; operator=(const MultiplexDenseOp&amp; other) = delete;
  ~MultiplexDenseOp() override = default;</p>
<p>void Compute(OpKernelContext* ctx) override {
    // Optional error checking: cond and a_values are lists of N, so there are
    // a total of 2N+1 inputs. Check that the  number of inputs and the
    // <code>N</code> Attr is consistent.
    const int64_t expected_inputs = 2 * num_cond_a_ + 1;
    OP_REQUIRES(ctx, expected_inputs == ctx-&gt;num_inputs(),
                Internal("expected_inputs != num_inputs(): ", expected_inputs,
                         " != ", ctx-&gt;num_inputs()));
    VLOG(1) &lt;&lt; "N " &lt;&lt; num_cond_a_;</p>
<pre><code>const auto&amp; first_cond_tensor = ctx-&gt;input(0);
const auto&amp; first_a_values_tensor = ctx-&gt;input(num_cond_a_);
const auto&amp; b_values_tensor = ctx-&gt;input(2 * num_cond_a_);

// Allow any shape, but require that a_values, b_values, and cond all
// have the same shape.
// Note that ::tensorflow::TensorShapeUtils has some useful functions
// for checking shapes.
for (int64_t i = 0; i &lt; num_cond_a_; i++) {
  const auto&amp; cond_tensor_i = ctx-&gt;input(i);
  const auto&amp; a_values_tensor_i = ctx-&gt;input(num_cond_a_ + i);
  OP_REQUIRES(
      ctx, a_values_tensor_i.shape() == b_values_tensor.shape(),
      InvalidArgument(
          "a_values[", i,
          "] and b_values must have the same shape. "
          "a_values[",
          i, "] shape: ", a_values_tensor_i.DebugString(),
          " b_values shape: ", b_values_tensor.shape().DebugString()));
  OP_REQUIRES(
      ctx, cond_tensor_i.shape() == b_values_tensor.shape(),
      InvalidArgument(
          "cond_values[", i,
          "] and b_valuesmust have the same shape. "
          "cond_values[",
          i, "] shape: ", first_a_values_tensor.shape().DebugString(),
          " b_values shape: ", first_cond_tensor.shape().DebugString()));
}

// Create an output tensor
Tensor* output_tensor = nullptr;
OP_REQUIRES_OK(
    ctx, ctx-&gt;allocate_output(0, b_values_tensor.shape(), &amp;output_tensor));
auto output = output_tensor-&gt;template flat&lt;T&gt;();

const auto b_values = b_values_tensor.template flat&lt;T&gt;();
// np.select style behavior, `cond` and `a_values` are lists of tensors.
// Also works for the np.where style case where there is only one `cond`
// and one `a_values` tensor.
const int64_t N = first_a_values_tensor.NumElements();
for (int64_t i = 0; i &lt; N; i++) {
  bool flag = false;
  for (int64_t list_index = 0; list_index &lt; num_cond_a_; list_index++) {
    const auto&amp; cond_tensor = ctx-&gt;input(list_index);
    const auto&amp; a_values_tensor = ctx-&gt;input(num_cond_a_ + list_index);
    const auto cond = cond_tensor.template flat&lt;bool&gt;();
    const auto a_values = a_values_tensor.template flat&lt;T&gt;();
    if (cond(i)) {
      output(i) = a_values(i);
      flag = true;
      VLOG(1) &lt;&lt; "A " &lt;&lt; list_index &lt;&lt; " for " &lt;&lt; i;
      break;
    }
  }
  if (!flag) {
    output(i) = b_values(i);
    VLOG(1) &lt;&lt; "B for " &lt;&lt; i;
  }
}
</code></pre>
<p>}</p>
<p>private:
  int64_t num_cond_a_;  // the number of <code>cond</code> and <code>a</code> input tensors
};
```</p>
<p>The kernel uses a private member variable (<code>num_cond_a_</code>) to hold the length of
<code>cond</code> and <code>a</code>. The constructor saves the <code>N</code> attribute into the variable.</p>
<!-- test_snippets_in_readme skip -->
<p><code>c++
private:
  int64_t num_cond_a_;  // the number of cond and a input tensors</code></p>
<!-- test_snippets_in_readme skip -->
<p><code>c++
explicit MultiplexDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx-&gt;GetAttr("N", &amp;num_cond_a_));
}</code></p>
<p>The <code>num_cond_a_</code> variable is used to index the inputs in the following order:
<code>cond</code>, <code>a</code>, <code>b</code>. The op interfaces specify that <code>cond</code> and <code>a</code> are tensor lists
of length <code>N</code>, and <code>b</code> is a single tensor. The inputs are indexed as follows:</p>
<ol>
<li><code>cond</code>: [0 ... N-1]</li>
<li><code>a</code>: [N ... 2*N-1]</li>
<li><code>b</code>: [2*N]</li>
</ol>
<p>When <code>num_cond_a_</code> is equal to 1, the kernel implements <code>numpy.where</code> as it
would in the <code>multiplex_2</code> op. When <code>num_cond_a_</code> is greater than 1, the kernel
implements <code>numpy.select</code>. This is achieved with the following <code>for</code> loop.</p>
<!-- test_snippets_in_readme skip -->
<p><code>c++
for (int64_t i = 0; i &lt; N; i++) {
  bool flag = false;
  for (int64_t list_index = 0; list_index &lt; num_cond_a_; list_index++) {
    const auto&amp; cond_tensor = ctx-&gt;input(list_index);
    const auto&amp; a_values_tensor = ctx-&gt;input(num_cond_a_ + list_index);
    const auto cond = cond_tensor.flat&lt;bool&gt;();
    const auto a_values = a_values_tensor.flat&lt;T&gt;();
    if (cond(i)) {
      output(i) = a_values(i);
      flag = true;
      break;
    }
  }
  if (!flag) {
    output(i) = b_values(i);
  }
}</code></p>
<h4>Compile the op</h4>
<p>Compile the C++ op to create a kernel library and Python wrapper that enables
you to use the op with TensorFlow.</p>
<p>Create a <code>BUILD</code> file for the op which declares the dependencies and the output
build targets. Refer to
<a href="https://www.tensorflow.org/guide/create_op#build_the_op_library">building for OSS</a>.</p>
<h3>Step 4 - Create the Python wrapper</h3>
<p>To create the Python wrapper, import and implement a function that serves as the
op's public API and provides a docstring.</p>
<p>If <code>cond</code> and <code>a</code> are not already lists, the wrapper in <code>multiplex_4_op.py</code>
puts the variables in lists before the <code>numpy.where</code> implementation.</p>
<p>Note: The generated Python wrapper automatically sets the <code>N</code> attribute based on
the length of the input lists.</p>
<p><code>``
def multiplex(cond, a, b, name=None):
  """Return elements chosen from</code>a<code>or</code>b<code>depending on</code>cond`.</p>
<p>This is similar to <code>np.where</code> and <code>tf.where</code> if <code>cond</code> and <code>a</code> are tensors.
  This is similar to <code>np.select</code> if <code>cond</code> and <code>a</code> are lists of tensors.
  In either case, this is simplified to only handle the case of dense tensors,
  no optional parameters, no broadcasting, etc..</p>
<blockquote>
<blockquote>
<blockquote>
<p>multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])
  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  1, 200, 300,   4], ...)></p>
<p>a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
cond1 = tf.constant([False, False, True, False, False], dtype=bool)
cond2 = tf.constant([False, False, False, False, True], dtype=bool)
cond3 = tf.constant([True, False, True, False, True], dtype=bool)
multiplex_4_op.multiplex([cond1, cond2, cond3], [a1, a2, a3], b)
  <tf.Tensor: shape=(5,), ... numpy=array([ 11, 102,   3, 104,  10], ...)></p>
</blockquote>
</blockquote>
</blockquote>
<p>Args:
    cond: tf.Tensor or list of tf.Tensor of type bool. Where True, yield <code>a</code>.
      When muliple corresponding <code>cond</code> elements are true, the first one yield
      based on the first one encountered.
    a: tf.Tensor or list of tf.Tensor, each with the same type and shape as <code>b</code>.
    b: tf.Tensor or list of tf.Tensor with the same type and shape as <code>a</code>. Yield
      <code>b</code> if all corresponding <code>cond</code> values is False.
    name: An optional name for the op.</p>
<p>Returns:
    A tf.Tensor with elements from <code>a</code> where <code>cond</code> is True, and elements
    from <code>b</code> elsewhere.
  """
  if not isinstance(cond, (list, tuple)):
    # Support "old" use of multiplex where <code>cond</code> and <code>a</code> are tensors,
    # not lists of tensors.
    return gen_multiplex_4_op.examples_multiplex_dense(
        cond=[cond], a_values=[a], b_values=b, name=name)
  return gen_multiplex_4_op.examples_multiplex_dense(
      cond=cond, a_values=a, b_values=b, name=name)
```</p>
<h3>Step 5 - Test the op</h3>
<p>Create op tests using classes derived from
<a href="https://www.tensorflow.org/api_docs/python/tf/test/TestCase"><code>tf.test.TestCase</code></a>.</p>
<p>When writing tests to ensure that the op works correctly in both graph and eager
executions, it is important to note that errors in the op code may be detected
in two distinct phases of code execution depending on how it is executed (eager
or graph executions). Errors may be detected early by the shape function or a
bit later from the logic in the <code>Compute</code> method. This may lead to differing
error types and messages.</p>
<p>```
@test_util.with_eager_op_as_function
class MultiplexOpTest(tf.test.TestCase):</p>
<p>@test_util.run_in_graph_and_eager_modes
  def test_multiplex_int(self):
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [1, 20, 3, 40, 5]
    result = multiplex_4_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)</p>
<p>@test_util.run_in_graph_and_eager_modes
  def test_multiplex_select(self):
    a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
    a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
    a = [a1, a2, a3]
    b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
    cond1 = tf.constant([False, False, True, False, False], dtype=bool)
    cond2 = tf.constant([False, False, False, False, True], dtype=bool)
    cond3 = tf.constant([True, False, True, False, True], dtype=bool)
    cond = [cond1, cond2, cond3]
    expect = np.select([self.evaluate(i) for i in cond],
                       [self.evaluate(i) for i in a], self.evaluate(b))
    # expected result is [11, 102, 3, 104, 10]
    result = multiplex_4_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)</p>
<p>def test_multiplex_saved_model(self):
    path = os.path.join(self.create_tempdir(), 'model')
    model_using_multiplex.save(multiplex_4_op.multiplex, path)
    result = model_using_multiplex.load_and_use(path)
    self.assertAllEqual(result, tf.constant([1, 20, 3, 40, 5], dtype=tf.int64))</p>
<p># One tf.function that uses both multiplex with single tensors for <code>cond</code>
  # and <code>a</code> and with lists of tensors for <code>cond</code> and <code>a</code>, i.e. a graph
  # with two example_multiplex_dense kernels that have different numbers
  # of inputs.
  @tf.function
  def _both(self):
    a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
    a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
    a_123 = [a1, a2, a3]
    b_123 = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
    cond1 = tf.constant([False, False, True, False, False], dtype=bool)
    cond2 = tf.constant([False, False, False, False, True], dtype=bool)
    cond3 = tf.constant([True, False, True, False, True], dtype=bool)
    cond_123 = [cond1, cond2, cond3]
    mux_123 = multiplex_4_op.multiplex(cond_123, a_123, b_123)
    b4 = tf.constant([201, 202, 203, 204, 205], dtype=tf.int64)
    cond4 = tf.constant([True, True, True, False, False], dtype=bool)
    result = multiplex_4_op.multiplex(cond4, mux_123, b4)
    return result</p>
<p>def test_both_single_and_list(self):
    result = self._both()
    self.assertAllEqual(result,
                        tf.constant([11, 102, 3, 204, 205], dtype=tf.int64))</p>
<p>@test_util.run_in_graph_and_eager_modes
  def test_inconsistent_inputs_error(self):
    a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
    a = [a1, a2]
    b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
    cond = tf.constant([False, False, True, False, False], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(a_values[0] and b_values must have the same shape'
        r')|('
        # Graph mode raises ValueError with the following message
        r'Shapes must be equal rank, but are 2 and 1)'):
      self.evaluate(multiplex_4_op.multiplex(cond, a, b))
```</p>
<p>The following <code>tf.function</code> in muliplex_4_test.py has two multiplex custom ops:
one that takes lists for its <code>cond</code> and <code>a</code> inputs, and another that takes
single tensors.</p>
<p><code>@tf.function
  def _both(self):
    a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
    a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
    a_123 = [a1, a2, a3]
    b_123 = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
    cond1 = tf.constant([False, False, True, False, False], dtype=bool)
    cond2 = tf.constant([False, False, False, False, True], dtype=bool)
    cond3 = tf.constant([True, False, True, False, True], dtype=bool)
    cond_123 = [cond1, cond2, cond3]
    mux_123 = multiplex_4_op.multiplex(cond_123, a_123, b_123)
    b4 = tf.constant([201, 202, 203, 204, 205], dtype=tf.int64)
    cond4 = tf.constant([True, True, True, False, False], dtype=bool)
    result = multiplex_4_op.multiplex(cond4, mux_123, b4)
    return result</code></p>
<p>The model_using_multiplex.py file has functions for creating and using a saved
custom op model <code>SavedModel</code>. In this test, the <code>multiplex_4</code> op is used to both
save and use models.</p>
<p><code>def test_multiplex_saved_model(self):
    path = os.path.join(self.create_tempdir(), 'model')
    model_using_multiplex.save(multiplex_4_op.multiplex, path)
    result = model_using_multiplex.load_and_use(path)
    self.assertAllEqual(result, tf.constant([1, 20, 3, 40, 5], dtype=tf.int64))</code></p>
<p>Test the op with the following:</p>
<!-- test_snippets_in_readme skip -->
<p><code>shell
bazel test //third_party/tensorflow/google/g3doc/example/multiplex_4:multiplex_4_test</code></p>
<p>Reuse the <code>BUILD</code> file to add build rules for the Python API wrapper and the op
test.</p>
<p><code>py_strict_library(
    name = "multiplex_4_op",
    srcs = ["multiplex_4_op.py"],
    data = ["multiplex_4_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)</code></p>
<!-- test_snippets_in_readme skip -->
<p><code>tf_py_test(
    name = "multiplex_4_test",
    size = "medium",  # This test blocks because it writes and reads a file,
    timeout = "short",  # but it still runs quickly.
    srcs = ["multiplex_4_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",
        "no_pip",
    ],
    deps = [
        ":model_using_multiplex",
        ":multiplex_4_op",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python/framework:errors",
        "//third_party/tensorflow/python/framework:test_lib",
    ],
)</code></p>
<h3>Use the op</h3>
<p>Build the op with the following:</p>
<!-- test_snippets_in_readme skip -->
<p><code>shell
bazel build //third_party/tensorflow/examples/custom_ops_doc/multiplex_4:multiplex_4_op</code></p>
<p>Import the op and call it using the following example:</p>
<!-- test_snippets_in_readme skip -->
<p>```python
import tensorflow as tf</p>
<p>from tensorflow.examples.custom_ops_doc.multiplex_4 import multiplex_4_op</p>
<p>a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
a = [a1, a2, a3]
b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
cond1 = tf.constant([False, False, True, False, False], dtype=bool)
cond2 = tf.constant([False, False, False, False, True], dtype=bool)
cond3 = tf.constant([True, False, True, False, True], dtype=bool)
cond = [cond1, cond2, cond3]</p>
<h1>expected result is [11, 102, 3, 104, 10]</h1>
<p>result = multiplex_4_op.multiplex(cond, a, b)
```</p>
<p>The <code>multiplex_4_load_use.py</code> file uses <code>load_and_use</code> from
<code>model_using_muliplex.py</code> to load a saved model from a <code>multiplex_2</code> op. The
saved model can be executed using the new kernel, (<code>multiplex_4</code>), which
supports both lists of tensors and single tensors for <code>cond</code> and <code>a</code> inputs.</p>
<p>Since <code>Examples&gt;MultiplexDense</code> can only be defined once in a binary, there must
be two separate binaries. A binary can either depend on <code>multiplex_2_op</code> or
<code>multiplex_4_op</code>, but not both. The custom ops are backward compatible, so we
can use <code>save</code> on <code>multiplex_2</code> and <code>load_and_use</code> on <code>multiplex_4</code>.</p>
<h3>Summary</h3>
<p>In this example, you learned how to implement a new multiplexer kernel that is
backwards compatible with an existing multiplexer kernel. This custom op handles
inputs that are lists of tensors, while continuing to handle inputs of single
tensors.</p>
<p>The tables below summarize the build rules and targets for building and testing
the <code>multiplex_4</code> op.</p>
<h4>Kernel components</h4>
<p>Op components                           | Build rule             | Build target         | Source
--------------------------------------- | ---------------------- | -------------------- | ------
Kernels (C++)                           | <code>tf_custom_op_library</code> | <code>multiplex_4_kernel</code> | <code>multiplex_4_kernel.cc</code>, <code>multiplex_4_op.cc</code>
Wrapper (automatically generated)       | N/A                    | <code>gen_multiplex_4_op</code> | N/A
Wrapper (with public API and docstring) | <code>py_strict_library</code>    | <code>multiplex_4_op</code>     | <code>multiplex_4_op.py</code>
Tests                                   | <code>tf_py_test</code>           | <code>multiplex_4_test</code>   | <code>multiplex_4_test.py</code></p>
<h5>Usage example</h5>
<p>Op components            | Build rule          | Build target               | Source
------------------------ | ------------------- | -------------------------- | ------
Common library           | <code>py_strict_library</code> | <code>model_using_multiplex</code>    | <code>model_using_multiplex.py</code>
Old op (with SavedModel) | <code>py_strict_binary</code>  | <code>multiplex_2_save</code>         | <code>multiplex_2_save.py</code>
New op (with SavedModel) | <code>py_strict_binary</code>  | <code>multiplex_4_load_and_use</code> | <code>multiplex_4_load_and_use.py</code></p>
<h2>Resources</h2>
<ul>
<li><a href="https://www.tensorflow.org/guide/create_op">OSS custom ops guide</a></li>
<li><a href="https://www.tensorflow.org/guide/saved_model">SavedModel</a></li>
<li><a href="https://numpy.org/doc/stable/reference/generated/numpy.select.html">Numpy Select</a></li>
</ul>
<!-- LINT.ThenChange(multiplex_4.md) -->