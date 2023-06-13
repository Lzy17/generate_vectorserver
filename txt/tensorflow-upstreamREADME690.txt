<!-- LINT.IfChange -->
<h1>Create a custom multiplexer op with dispatch to special case kernels</h1>
<p>This guide provides an end-to-end example of handling special cases with a new
C++ kernel. The custom op includes a Python wrapper that uses
<a href="https://www.tensorflow.org/guide/extension_type#tensor_api_dispatch">dispatch decorators</a>
to override the default behavior of TensorFlow operations when applied to
tensor-like types. For more information, refer to
<a href="https://www.tensorflow.org/guide/extension_type">extension types</a>.</p>
<p>Special case kernels can add new functionality to an existing op without any
required changes to existing kernels that have already been registered. For
example, a special case kernel can enable an existing op to handle a different
type of input.</p>
<p>Optional Python wrappers can enable a variety of non-breaking future changes,
though it is important to avoid any non-TensorFlow Python code in the
implementation. This is because any non-Tensorflow Python code will only be used
in eager execution and not in
<a href="https://www.tensorflow.org/api_docs/python/tf/function"><code>tf.function</code></a>
execution.</p>
<p>Python wrappers can serve the following purposes:</p>
<ul>
<li>
<p><strong>Handling special cases</strong>: The default C++ kernel handles normal cases,
    while another C++ kernel handles special cases. For example,
    <a href="https://www.tensorflow.org/api_docs/python/tf/math/add"><code>tf.add</code></a> calls one
    of two different C++ kernels, depending on whether the inputs are strings or
    not.</p>
</li>
<li>
<p><strong>Decomposing operations into multiple kernels</strong>: Some operations at the
    Python level are decomposed into multiple C++ kernels, rather than
    implemented through a single kernel. For example, there is no
    <code>ReduceVariance</code> kernel for the
    <a href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_variance"><code>tf.reduce_variance</code></a>
    op. Instead, the Python <code>reduce_variance</code> function computes the variance
    based on squared deviations from the mean.</p>
</li>
<li>
<p><strong>Adding an argument to an existing op</strong>: Since <code>name</code> is always the last
    argument wrapper, adding a new, optional argument requires a new wrapper.
    This prevents the op from mistaking the new argument for the <code>name</code>
    argument.</p>
</li>
<li>
<p><strong>Changing the order of arguments</strong>: Similar to the point above, a wrapper
    can be used to change the order of arguments.</p>
</li>
</ul>
<p>The content on this page assumes familiarity with the high-level process for
adding custom ops to TensorFlow. For additional context, read the
<a href="https://www.tensorflow.org/guide/create_op">OSS guide on creating custom ops</a>.</p>
<h2>Dispatch to special case kernels using sparse tensors</h2>
<p>This example demonstrates how you can create a Python custom multiplexer,
<code>multiplex_3_op</code>, to register a new kernel. If an existing op already handles
certain kinds of inputs, a special case kernel can extend the op to handle a
different kind of input without changing the existing op.</p>
<p>The special kernel (<code>multiplex_3</code>) in this example extends an existing op
(<code>multiplex_2</code>) so that it can handle
<a href="https://www.tensorflow.org/guide/sparse_tensor">sparse tensors</a> as inputs. This
provides the custom op with the following two kernels:</p>
<ul>
<li><strong>Default kernel</strong>: registers the multiplex op with dense tensors
    (<code>MultiplexDense</code>).</li>
<li><strong>Special case kernel</strong>: registers the multiplex op with sparse tensors
    (<code>MultiplexSparse</code>).</li>
</ul>
<p>The sparse tensor object (<code>tf.SparseTensor</code>) is appropriate for tensors that
contain missing values. Storing sparse values in sparse tensors is more
memory-efficient than storing in a dense tensor.</p>
<p>In this example, the default kernel is the
<code>multiplex_2</code> (multiplex_2_kernel.cc) kernel, and the new kernel
(multiplex_3_kernel.cc) implements the multiplex with sparse tensors.</p>
<p>Like other multiplex custom ops, <code>multiplex_3</code> is similar to
<a href="https://tensorflow.org/api_docs/python/tf/where?version=nightly"><code>tf.where</code></a>.
It returns elements chosen from either of the two input tensors (<code>x</code> or <code>y</code>),
depending on the <code>condition</code>. You can call the op with the following:</p>
<!-- test_snippets_in_readme skip -->
<p><code>python
multiplex_3_op.multiplex(condition, x, y)</code></p>
<p>This simplified <code>multiplex_3</code> op has the following limitations that are not
present in <code>tf.where</code>:</p>
<ul>
<li>Support only for CPU computations</li>
<li>No broadcasting capabilities</li>
<li>No extensibility through optional parameters</li>
</ul>
<p>This example contains C++ and Python code snippets to illustrate the code flow.
These snippets may be missing namespace declarations, imports, and test cases.</p>
<h3>Step 1 - Define the op interface</h3>
<p>Define the op interface and register it using the <code>REGISTER_OP</code> macro.</p>
<p><code>``
REGISTER_OP("Examples&gt;MultiplexSparse")
    .Input("cond_indices: int64")
    .Input("cond_values: bool")
    .Input("cond_shape: int64")
    .Input("a_indices: int64")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b_indices: int64")
    .Input("b_values: T")
    .Input("b_shape: int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("T: type")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c-&gt;WithRank(c-&gt;input(0), 2, &amp;unused));  // cond_indices
      TF_RETURN_IF_ERROR(c-&gt;WithRank(c-&gt;input(1), 1, &amp;unused));  // cond_values
      TF_RETURN_IF_ERROR(c-&gt;WithRank(c-&gt;input(2), 1, &amp;unused));  // cond_shape
      TF_RETURN_IF_ERROR(c-&gt;WithRank(c-&gt;input(3), 2, &amp;unused));  // a_indices
      TF_RETURN_IF_ERROR(c-&gt;WithRank(c-&gt;input(4), 1, &amp;unused));  // a_values
      TF_RETURN_IF_ERROR(c-&gt;WithRank(c-&gt;input(5), 1, &amp;unused));  // a_shape
      TF_RETURN_IF_ERROR(c-&gt;WithRank(c-&gt;input(6), 2, &amp;unused));  // b_indices
      TF_RETURN_IF_ERROR(c-&gt;WithRank(c-&gt;input(7), 1, &amp;unused));  // b_values
      TF_RETURN_IF_ERROR(c-&gt;WithRank(c-&gt;input(8), 1, &amp;unused));  // b_shape
      const auto num_rows = c-&gt;UnknownDim();
      const auto dense_rank = c-&gt;UnknownDim();
      c-&gt;set_output(0, c-&gt;Matrix(num_rows, dense_rank));
      c-&gt;set_output(1, c-&gt;Vector(num_rows));
      c-&gt;set_output(2, c-&gt;Vector(dense_rank));
      return ::tensorflow::OkStatus();
    })
    .Doc(R"doc(
Return elements chosen from</code>a<code>or</code>b<code>depending on</code>cond`.</p>
<p>This is similar to <code>np.where</code> and <code>tf.where</code>, but simplified to only handle
the case of sparse tensors that are vectors, no optional parameters,
no broadcasting, etc.. Elements for <code>a</code> are chosen if there is a <code>true</code> <code>cond</code>
value at the same position. Elements for <code>b</code> are chosen if there is not a <code>true</code>
<code>cond</code> value at the same position, i.e., if either there is a <code>false</code> <code>cond</code>
value or the <code>cond</code> value is not specified.</p>
<p>Indices must be ordered as described by tf.sparse_reorder.</p>
<p>cond_indices: a rank-2 tensor of sparse indices.
cond_values: a rank-1 tensor of sparse values.
cond_shape: a rank-1 tensor representing the dense shape.
a_indices: a rank-2 tensor of sparse indices.
a_values: a rank-1 tensor of sparse values.
a_shape: a rank-1 tensor representing the dense shape.
b_indices: a rank-2 tensor of sparse indices.
b_values: a rank-1 tensor of sparse values.
b_shape: a rank-1 tensor representing the dense shape.
output_indices: a rank-2 tensor of sparse indices.
output_values: a rank-1 tensor of sparse values.
output_shape: a rank-1 tensor representing the dense shape.
)doc");
```</p>
<h4>Inputs and outputs</h4>
<p>This op contains a total of nine input tensors. This is made up of three sparse
tensors (<code>a</code>, <code>b</code>, and <code>cond</code>), where each sparse tensor is encoded using the
coordinate list (COO) format:</p>
<ul>
<li><code>values</code>: 1D tensor with shape <code>[N]</code> containing all non-zero values.</li>
<li><code>indices</code>: 2D tensor with shape <code>[N, rank]</code>, containing the indices of the
    non-zero values</li>
<li><code>dense_shape</code>: 1D tensor with shape <code>[rank]</code>, specifying the shape of the
    tensor.</li>
</ul>
<p>The <code>cond</code> tensor accepts a boolean value to select between <code>a</code> and <code>b</code>, and the
<code>a</code> and <code>b</code> tensors accept a value of type <code>T</code>. The output tensor also contains
a value of type <code>T</code>.</p>
<h4>Shape function</h4>
<p>Unlike dense tensors, which have a fixed shape, the shape of sparse tensors
depend on the number of non-missing values in the output. Since this can not be
determined by the shape of the inputs, the shape function (<code>SetShapeFn</code>) uses
<code>UnknownDim()</code>.</p>
<!-- test_snippets_in_readme skip -->
<p><code>c++
  .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      // Error checking omitted, see source file.
      const auto num_rows = c-&gt;UnknownDim();
      const auto dense_rank = c-&gt;UnknownDim();
      c-&gt;set_output(0, c-&gt;Matrix(num_rows, dense_rank));
      c-&gt;set_output(1, c-&gt;Vector(num_rows));
      c-&gt;set_output(2, c-&gt;Vector(dense_rank));
      return tensorflow::OkStatus();
    })</code></p>
<h4>Attributes and docstrings</h4>
<p>The <code>Attr</code> for this op is defined as <code>.Attr("T: type")</code>, which specifies <code>T</code> as
an <code>Attr</code> of type <code>type</code>. In the subsequent steps, you will use <code>T</code> with a
template class to define the type of the contents of tensors.</p>
<p>The docstring for this op is specified by passing a string to <code>.Doc()</code>.</p>
<h3>Step 2 - Register the op implementation (kernel)</h3>
<p>The C++ kernel in <code>multiplex_3_kernel.cc</code> implements a multiplex for sparse
tensors. For simplicity, this example only supports rank 1 sparse tensors
(sparse vectors).</p>
<p>Register the kernel by calling the <code>REGISTER_KERNEL_BUILDER</code> macro.</p>
<p>```
#define REGISTER_KERNELS_CPU(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Examples&gt;MultiplexSparse")      \
                              .Device(::tensorflow::DEVICE_CPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexSparseOp<type>)
TF_CALL_ALL_TYPES(REGISTER_KERNELS_CPU);</p>
<h1>undef REGISTER_KERNELS_CPU</h1>
<p>```</p>
<h3>Step 3 - Implement the op kernel(s)</h3>
<p>In the <code>multiplex_3_kernel.cc</code> op kernel, create a class derived from
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#:~:text=class%20OpKernel"><code>OpKernel</code></a>
that implements a <code>Compute</code> method. This method retrieves and validates input
tensors, performs computation, and creates output tensors.</p>
<p>```
template <typename T>
class MultiplexSparseOp : public OpKernel {
 public:
  explicit MultiplexSparseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  MultiplexSparseOp(const MultiplexSparseOp&amp; other) = delete;
  MultiplexSparseOp&amp; operator=(const MultiplexSparseOp&amp; other) = delete;
  ~MultiplexSparseOp() override = default;</p>
<p>void Compute(OpKernelContext* ctx) override {
    const auto&amp; cond_indices_tensor = ctx-&gt;input(0);
    const auto&amp; cond_values_tensor = ctx-&gt;input(1);
    const auto&amp; cond_shape_tensor = ctx-&gt;input(2);
    const auto&amp; a_indices_tensor = ctx-&gt;input(3);
    const auto&amp; a_values_tensor = ctx-&gt;input(4);
    const auto&amp; a_shape_tensor = ctx-&gt;input(5);
    const auto&amp; b_indices_tensor = ctx-&gt;input(6);
    const auto&amp; b_values_tensor = ctx-&gt;input(7);
    const auto&amp; b_shape_tensor = ctx-&gt;input(8);
    OP_REQUIRES_OK(ctx,
                   ValidateSparseTensor(cond_indices_tensor, cond_values_tensor,
                                        cond_shape_tensor, "cond"));
    OP_REQUIRES_OK(ctx, ValidateSparseTensor(a_indices_tensor, a_values_tensor,
                                             a_shape_tensor, "a"));
    OP_REQUIRES_OK(ctx, ValidateSparseTensor(b_indices_tensor, b_values_tensor,
                                             b_shape_tensor, "b"));
    OP_REQUIRES(
        ctx, cond_shape_tensor.shape() == a_shape_tensor.shape(),
        InvalidArgument("Sparse tensors must be the same shape. cond_shape: ",
                        cond_shape_tensor.shape().DebugString(),
                        " vs a_shape: ", a_shape_tensor.shape().DebugString()));
    OP_REQUIRES(
        ctx, a_shape_tensor.shape() == b_shape_tensor.shape(),
        InvalidArgument("Sparse tensors must be the same shape. a_shape: ",
                        a_shape_tensor.shape().DebugString(),
                        " vs b_shape: ", b_shape_tensor.shape().DebugString()));
    const int rank = a_shape_tensor.dim_size(0);
    OP_REQUIRES(
        ctx, rank == 1,
        InvalidArgument("Sorry, multiplex for sparse tensors only "
                        "supports rank 1 tensors to simplify this example."));
    const int cond_elements = cond_indices_tensor.dim_size(0);
    const int a_elements = a_indices_tensor.dim_size(0);
    const int b_elements = b_indices_tensor.dim_size(0);
    const auto cond_indices = cond_indices_tensor.matrix<int64_t>();
    const auto cond_values = cond_values_tensor.flat<bool>();
    const auto cond_shape = cond_shape_tensor.flat<int64_t>();
    const auto a_indices = a_indices_tensor.matrix<int64_t>();
    const auto a_values = a_values_tensor.flat<T>();
    const auto a_shape = a_shape_tensor.flat<int64_t>();
    const auto b_indices = b_indices_tensor.matrix<int64_t>();
    const auto b_values = b_values_tensor.flat<T>();
    const auto b_shape = b_shape_tensor.flat<int64_t>();
    int cond_index = 0;
    int a_index = 0;
    int b_index = 0;
    // This vector is a list of source tensors (a = true, b = false) and source
    // indices.
    std::vector<std::pair\<bool, int>> merged_output;
    merged_output.reserve(std::min(cond_elements, a_elements) + b_elements);
    while (a_index &lt; a_elements || b_index &lt; b_elements) {
      // Determine the whether the current location with values has a value
      // for <code>a</code>, for <code>b</code> or for both <code>a</code> and <code>b</code>.
      int64_t cur_row;
      bool is_a_at_cur = false;
      bool is_b_at_cur = false;
      if (a_index &lt; a_elements &amp;&amp; b_index &lt; b_elements) {
        const int64_t a_row = a_indices(a_index, 0);
        const int64_t b_row = b_indices(b_index, 0);
        cur_row = std::min(a_row, b_row);
        if (a_row == cur_row) {
          is_a_at_cur = true;
        }
        if (b_row == cur_row) {
          is_b_at_cur = true;
        }
      } else if (a_index &lt; a_elements) {
        cur_row = a_indices(a_index, 0);
        is_a_at_cur = true;
      } else {  // b_index &lt; b_elements
        cur_row = b_indices(b_index, 0);
        is_b_at_cur = true;
      }
      // Deterimine if <code>cond</code> has a value at the current location
      bool cond_flag = false;
      while (cond_index &lt; cond_elements) {
        const int64_t cond_row = cond_indices(cond_index, 0);
        if (cond_row &gt; cur_row) {
          break;
        }
        if (cond_row == cur_row) {
          cond_flag = cond_values(cond_index);
          break;
        }
        ++cond_index;
      }
      // Add <code>a</code> or <code>b</code> to the merged output based on the condition
      if (is_a_at_cur) {
        if (cond_flag) {
          merged_output.emplace_back(true, a_index);
        }
        ++a_index;
      }
      if (is_b_at_cur) {
        if (!cond_flag) {
          merged_output.emplace_back(false, b_index);
        }
        ++b_index;
      }
    }</p>
<pre><code>// Allocate output tensors.
Tensor* output_indices_tensor;
Tensor* output_values_tensor;
Tensor* output_dense_shape_tensor;
const int num_values = merged_output.size();
OP_REQUIRES_OK(ctx, ctx-&gt;allocate_output(0, TensorShape({num_values, rank}),
                                         &amp;output_indices_tensor));
OP_REQUIRES_OK(ctx, ctx-&gt;allocate_output(1, TensorShape({num_values}),
                                         &amp;output_values_tensor));
OP_REQUIRES_OK(ctx, ctx-&gt;allocate_output(2, TensorShape({rank}),
                                         &amp;output_dense_shape_tensor));
auto output_indices = output_indices_tensor-&gt;matrix&lt;int64_t&gt;();
auto output_values = output_values_tensor-&gt;flat&lt;T&gt;();
auto output_shape = output_dense_shape_tensor-&gt;flat&lt;int64_t&gt;();
for (int row = 0; row &lt; num_values; ++row) {
  const auto&amp; source_flag = merged_output[row].first;
  const auto&amp; source_row = merged_output[row].second;
  const auto&amp; indices = source_flag ? a_indices : b_indices;
  const auto&amp; values = source_flag ? a_values : b_values;
  for (int column = 0; column &lt; rank; ++column) {
    output_indices(row, column) = indices(source_row, column);
  }
  output_values(row) = values(source_row);
}
// Expand the shape of the output sparse tensor so that it is as large
// as the shape of the largest input in each dimension.
// An alternative behavoir would be to require that the shapes be the
// same and implement error checking that all the corresponding values
// in the shape tensors are the same (e.g.
// `cond_shape(i) == a_shape(i)` and `a_shape(i) == b_shape(i)` in
// OP_REQUIRES above and `output_shape(i) = a_shape(i)` here).
for (int i = 0; i &lt; rank; ++i) {
  output_shape(i) =
      std::max(cond_shape(i), std::max(a_shape(i), b_shape(i)));
}
</code></pre>
<p>}</p>
<p>private:
  Status ValidateSparseTensor(const ::tensorflow::Tensor&amp; indices_tensor,
                              const ::tensorflow::Tensor&amp; values_tensor,
                              const ::tensorflow::Tensor&amp; shape_tensor,
                              const string label) {
    if (!TensorShapeUtils::IsMatrix(indices_tensor.shape())) {
      return InvalidArgument(
          "Sparse indices for ", label,
          " must be rank 2, not shape: ", indices_tensor.shape().DebugString());
    }
    if (!TensorShapeUtils::IsVector(values_tensor.shape())) {
      return InvalidArgument("Sparse values for ", label,
                             " must be a vector, not shape: ",
                             values_tensor.shape().DebugString());
    }
    if (!TensorShapeUtils::IsVector(shape_tensor.shape())) {
      return InvalidArgument(
          "Sparse shape for ", label,
          " must be a vector, not shape: ", shape_tensor.shape().DebugString());
    }
    if (indices_tensor.dim_size(0) != values_tensor.dim_size(0)) {
      return InvalidArgument("Sparse indices and values for " + label +
                                 " must have the same "
                                 "number of rows. indices: ",
                             indices_tensor.shape().DebugString(),
                             " values: ", values_tensor.shape().DebugString());
    }
    return OkStatus();
  }
};
```</p>
<p>The following snippet shows how the kernel accesses one of the inputs.</p>
<!-- test_snippets_in_readme skip -->
<p><code>c++
  void Compute(OpKernelContext* ctx) override {
    const auto&amp; cond_indices_tensor = ctx-&gt;input(0);
    const auto&amp; cond_values_tensor = ctx-&gt;input(1);
    const auto&amp; cond_shape_tensor = ctx-&gt;input(2);
    // Error checking omitted, see source file.
    const int cond_elements = cond_indices_tensor.dim_size(0);
    const auto cond_indices = cond_indices_tensor.matrix&lt;int64_t&gt;();
    const auto cond_values = cond_values_tensor.flat&lt;bool&gt;();
    const auto cond_shape = cond_shape_tensor.flat&lt;int64_t&gt;();
  }</code></p>
<p>The kernel implements the multiplex operation through the following steps:</p>
<ol>
<li>
<p><strong>Create an empty list</strong>: The list contains elements that refer to values of
    <code>a</code> or <code>b</code> in merged order. The elements are pairs of a <code>bool</code> to indicate
    the source tensor (a = true, b = false) and an <code>int</code> to indicate the source
    index.</p>
</li>
<li>
<p><strong>Append values in <code>a</code> and <code>b</code> to the list</strong>: Looping through all
    non-missing values in <code>a</code> and <code>b</code>, determine whether each position has a
    non-missing value in <code>a</code>, <code>b</code>, or both.</p>
<p>If <code>cond</code> is true and <code>a</code> has a non-missing value at that position, append
this element to the list. If <code>cond</code> is false or missing and <code>b</code> has a
non-missing value at that position, append this element to the list.</p>
<p>Note: Assume that indices are sorted in canonical row-major order (e.g.
using
<a href="https://www.tensorflow.org/api_docs/python/tf/sparse/reorder"><code>tf.sparse.reorder</code></a>).</p>
</li>
<li>
<p><strong>Allocate the output</strong>: The size of the output is based on the length of
    the list.</p>
</li>
<li>
<p><strong>Add indices and values of <code>a</code> and <code>b</code> to the output</strong>: Iterate through the
    elements in the list and copy the indices and values of <code>a</code> and <code>b</code> to the
    output.</p>
</li>
<li>
<p><strong>Set the shape of the output</strong>: Set the (dense) shape of the output to
    match the largest of the inputs. Shaping the output is accomplished with
    following snippet:</p>
<p><!-- test_snippets_in_readme skip -->
<code>c++
for (int i = 0; i &lt; rank; ++i) {
  output_shape(i) =
      std::max(cond_shape(i), std::max(a_shape(i), b_shape(i)));
}</code></p>
</li>
</ol>
<h5>Considerations when setting the output shape</h5>
<p>This implementation is specific to sparse tensors. The expansion is simple
because the concept of "missing values" for sparse tensors is well-defined.
There is no exact equivalent for dense tensors.</p>
<p>In many cases, sparse tensors are just sparse encodings of a dense tensor. In
these cases, all inputs should have the same dense shape, and the output shape
would be identical to the shape of the inputs.</p>
<p>In these cases, you can replace the <code>std::max calculation</code> with
<code>output_shape(i) = a_shape(i)</code> after verifying the following conditions in
<code>OP_REQUIRES</code>:</p>
<!-- test_snippets_in_readme skip -->
<p><code>c++
cond_shape(i) == a_shape(i)
a_shape(i) == b_shape(i)</code></p>
<h3>Step 4 - Create the Python wrapper</h3>
<p>To create the Python wrapper, import and implement a function that serves as the
op's public API and provides a docstring.</p>
<p>When the inputs are
<a href="https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor"><code>tf.SparseTensor</code></a>,
the
<a href="https://www.tensorflow.org/guide/extension_type#tensor_api_dispatch">dispatch decorator</a>
in the snippet below prompts the previous
<code>gen_multiplex_2_op.examples_multiplex_dense</code> op to use the new C++ kernel
wrapped by <code>gen_multiplex_3_op.examples_multiplex_sparse</code>.</p>
<p>Note: This optional Python wrapper depends on the <code>multiplex_2</code> op in addition
to the <code>multiplex_3</code> dependencies. See <code>deps</code> for <code>multiplex_3_op</code> in the BUILD
file.</p>
<p><code>``
@tf.experimental.dispatch_for_api(gen_multiplex_2_op.examples_multiplex_dense)
def multiplex_sparse(cond: tf.SparseTensor,
                     a: tf.SparseTensor,
                     b: tf.SparseTensor,
                     name=None):
  """Return elements chosen from</code>a<code>or</code>b<code>depending on</code>cond`.</p>
<p>This is similar to <code>np.where</code> and <code>tf.where</code>, but simplified to only handle
  the case of rank 1 sparse tensors, no optional parameters, no broadcasting,
  etc..</p>
<blockquote>
<blockquote>
<blockquote>
<p>cond = tf.SparseTensor(
  ...     indices=[[1], [3], [6]], values=[True, False, True], dense_shape=[7])
a = tf.sparse.from_dense(['', 'a0', '', 'a1', '', 'a2', ''])
b = tf.sparse.from_dense(['b0', '', 'b1', 'b2', '', '', 'b3'])
multiplex_3_op.multiplex_sparse(cond, a, b)
  SparseTensorValue(indices=array([[0],
    [1],
    [2],
    [3]]), values=array([b'b0', b'a0', b'b1', b'b2'], dtype=object),
    dense_shape=array([7]))
  Args:
    cond: tf.SparseTensor of type bool. Where True, yield <code>a</code>, otherwise yield
      <code>b</code>.
    a: tf.SparseTensor with the same type and shape as <code>b</code>.
    b: tf.SparseTensor with the same type and shape as <code>a</code>.
    name: An optional name for the op.</p>
</blockquote>
</blockquote>
</blockquote>
<p>Returns:
    A tf.SparseTensor with elements from <code>a</code> where <code>cond</code> is True, and elements
    from <code>b</code> elsewhere.
  """
  (indices, values, shape) = examples_multiplex_sparse(
      cond_indices=cond.indices,
      cond_values=cond.values,
      cond_shape=cond.dense_shape,
      a_indices=a.indices,
      a_values=a.values,
      a_shape=a.dense_shape,
      b_indices=b.indices,
      b_values=b.values,
      b_shape=b.dense_shape,
      name=name)
  return tf.SparseTensor(indices, values, shape)
```</p>
<h3>Step 5 - Test the op</h3>
<p>Create op tests using classes derived from
<a href="https://www.tensorflow.org/api_docs/python/tf/test/TestCase"><code>tf.test.TestCase</code></a>.</p>
<p>When writing tests to ensure that the op works correctly in both graph and eager
executions, it is important to note that errors in the op code may be detected
in two distinct phases of code execution depending on how it is executed (eager
or graph executions). Errors may be detected early by the shape function or a
bit later from the logic in the <code>Compute</code> method. This may lead to differing
error types and/or messages.</p>
<p>The following tests use the <code>multiplex_2_op.multiplex</code> custom op from the
previous <code>multiplex_2</code> example, which now supports sparse tensors (while
continuing to support dense tensors). The <code>test_sparse_op_different</code> test inputs
are sparse tensors, so it uses the new <code>multiplex_3_kernel</code> C++ kernel.</p>
<p><code>@test_util.run_in_graph_and_eager_modes
  def test_sparse_op_different(self):
    cond = tf.SparseTensor(
        indices=[[1], [3], [6]], values=[True, False, True], dense_shape=[7])
    a = tf.SparseTensor(
        indices=[[1], [3], [5]], values=['a0', 'a1', 'a2'], dense_shape=[6])
    b = tf.SparseTensor(
        indices=[[0], [2], [3], [6]],
        values=['b0', 'b1', 'b2', 'b3'],
        dense_shape=[7])
    result = self.evaluate(multiplex_2_op.multiplex(cond, a, b))
    self.assertAllEqual([7], result.dense_shape)
    self.assertAllEqual([[0], [1], [2], [3]], result.indices)
    self.assertAllEqual([b'b0', b'a0', b'b1', b'b2'], result.values)</code></p>
<p>The <code>test_multiplex_int</code> test inputs are dense tensors, so it uses the old
<code>multiplex_2_kernel</code> C++ kernel.</p>
<p><code>@test_util.run_in_graph_and_eager_modes
  def test_multiplex_int(self):
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [1, 20, 3, 40, 5]
    result = multiplex_2_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)</code></p>
<p>Refer to <code>multiplex_3_test.py</code> for the full source code which contains all the
test cases.</p>
<p>Reuse the <code>BUILD</code> file to add build rules for the Python API wrapper and the op
test.</p>
<p>```
tf_custom_op_library(
    name = "multiplex_3_kernel.so",
    srcs = [
        "multiplex_3_kernel.cc",
        "multiplex_3_op.cc",
    ],
)</p>
<p>py_strict_library(
    name = "multiplex_3_op",
    srcs = ["multiplex_3_op.py"],
    data = [":multiplex_3_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/examples/custom_ops_doc/multiplex_2:multiplex_2_op",
    ],
)
```</p>
<p>Test the op with the following:</p>
<!-- test_snippets_in_readme skip -->
<p><code>shell
bazel test //third_party/tensorflow/google/g3doc/example/multiplex_3:multiplex_3_test</code></p>
<h3>Use the op</h3>
<p>Import the op and call it using the following example:</p>
<!-- test_snippets_in_readme skip -->
<p>```python
import tensorflow as tf</p>
<p>from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op
from tensorflow.examples.custom_ops_doc.multiplex_3 import multiplex_3_op</p>
<p>cond = tf.SparseTensor(indices=[[1], [3], [6]],
                    values=[True, False, True], dense_shape=[7])
a = tf.SparseTensor(indices=[[1], [3], [5]],
                    values=['a0', 'a1', 'a2'], dense_shape=[6])
b = tf.SparseTensor(indices=[[0], [2], [3], [6]],
                    values=['b0', 'b1', 'b2', 'b3'], dense_shape=[7])
result = multiplex_2_op.multiplex(cond, a, b)
```</p>
<p>Here, <code>multiplex_2_op</code> is the name of the Python wrapper that was created in the
multiplex_2 example. Importing the <code>multiplex_3_op</code> Python wrapper created in
this example extends <code>multiplex_2_op.multiplex</code> to handle sparse tensors.</p>
<p>Build the op with the following:</p>
<!-- test_snippets_in_readme skip -->
<p><code>shell
bazel build //third_party/tensorflow/examples/custom_ops_doc/multiplex_3:multiplex_3_op</code></p>
<h3>Summary</h3>
<p>In this example, you learned how implement a new multiplexer kernel to handle
special cases. With a Python wrapper that uses
<a href="https://www.tensorflow.org/guide/extension_type#tensor_api_dispatch">dispatch decorators</a>
to override the default kernel, this custom op uses a new kernel to handle
sparse tensors.</p>
<p>The table below summarizes the build rules and targets for building and testing
the <code>multiplex_3</code> op.</p>
<p>Op components                           | Build rule             | Build target         | Source
--------------------------------------- | ---------------------- | -------------------- | ------
Kernels (C++)                           | <code>tf_custom_op_library</code> | <code>multiplex_3_kernel</code> | <code>multiplex_3_kernel.cc</code>, <code>multiplex_3_op.cc</code>
Wrapper (automatically generated)       | N/A                    | <code>gen_multiplex_3_op</code> | N/A
Wrapper (with public API and docstring) | <code>py_strict_library</code>    | <code>multiplex_3_op</code>     | <code>multiplex_3_op.py</code>
Tests                                   | <code>tf_py_test</code>           | <code>multiplex_3_test</code>   | <code>multiplex_3_test.py</code></p>
<h2>Resources</h2>
<ul>
<li><a href="https://www.tensorflow.org/guide/create_op">OSS custom ops guide</a></li>
<li><a href="https://www.tensorflow.org/guide/extension_type#tensor_api_dispatch">Extension types and dispatch decorators</a></li>
<li><a href="https://www.tensorflow.org/guide/sparse_tensor">Working with sparse tensors</a></li>
</ul>
<!-- LINT.ThenChange(multiplex_3.md) -->