<!-- LINT.IfChange -->
<h1>Create a custom multiplexer op</h1>
<p>This page provides an end-to-end example for adding a custom multiplexer op to
TensorFlow. For additional context,
read the
<a href="https://www.tensorflow.org/guide/create_op">OSS guide on creating custom ops</a>.</p>
<h2>Creating a custom multiplexer op</h2>
<p>This examples demonstrates how you can create a Python custom multiplexer
<code>multiplex_1_op</code>, similar to
<a href="https://tensorflow.org/api_docs/python/tf/where?version=nightly"><code>tf.where</code></a>
which you can call as:</p>
<!-- test_snippets_in_readme skip -->
<p><code>python
multiplex_1_op.multiplex(condition, x, y)                                        # doctest: skip</code></p>
<p>This custom op returns elements chosen from either of the two input tensors <code>x</code>
or <code>y</code> depending on the <code>condition</code>.</p>
<p>Example usage:</p>
<!-- test_snippets_in_readme skip -->
<p>```python
from tensorflow.examples.custom_ops_doc.multiplex_1 import multiplex_1_op</p>
<p>m = multiplex_1_op.multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])
m.numpy()
```</p>
<!-- test_snippets_in_readme skip -->
<p><code>array([  1, 200, 300,   4], dtype=int32)</code></p>
<p>Note that this simplified <code>multiplex_1</code> op has limitations that are not present
in <code>tf.where</code> such as:</p>
<ul>
<li>Support only for dense tensors</li>
<li>Support only for CPU computations</li>
<li>No broadcasting capabilities</li>
<li>No extensibility through optional parameters</li>
</ul>
<p>The example below contains C++ and Python code snippets to illustrate the code
flow. These snippets are not all complete; some are missing namespace
declarations, imports, and test cases.</p>
<h3>Step 1 - Define op interface</h3>
<p>Define the op interface and register it using the <code>REGISTER_OP</code> macro.</p>
<!-- test_snippets_in_readme skip -->
<p>```</p>
<h1>include "tensorflow/core/framework/op.h"</h1>
<h1>include "tensorflow/core/framework/shape_inference.h"</h1>
<p>```</p>
<p>```
REGISTER_OP("Examples1&gt;MultiplexDense")
    .Input("cond: bool")
    .Input("a_values: T")
    .Input("b_values: T")
    .Output("output_values: T")
    .Attr("T: type")
    .SetShapeFn(<a href="tensorflow::shape_inference::InferenceContext* c"></a> {
      // Determine the output shape and also assert that inputs 0 and 1 have
      // the same shape.
      tensorflow::shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c-&gt;Merge(c-&gt;input(0), c-&gt;input(1), &amp;out));
      // Assert that inputs 0 and 2 have the same shape, i.e. that all inputs
      // have the same shape. This is optional, but it is desirable
      // to raise errors about inconsistent input shapes early when using
      // graph mode.
      tensorflow::shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c-&gt;Merge(c-&gt;input(0), c-&gt;input(2), &amp;unused));</p>
<pre><code>  c-&gt;set_output(0, out);
  return ::tensorflow::OkStatus();
})
.Doc(R"doc(
</code></pre>
<p>Return elements chosen from <code>a</code> or <code>b</code> depending on <code>cond</code>.</p>
<p>This is similar to <code>np.where</code> and <code>tf.where</code>, but simplified to only handle
the case of dense tensors, no optional parameters, no broadcasting, etc..</p>
<p>cond: tf.Tensor of type bool.
a_values: tf.Tensor with the same type and shape as <code>b_values</code>.
b_values: tf.Tensor with the same type and shape as <code>a_values</code>.</p>
<pre><code>  Where True, yield `a_values`, otherwise yield `b_values`.
</code></pre>
<p>output_values: A tf.Tensor with elements from <code>a</code> where <code>cond</code> is True, and
               elements from <code>b</code> elsewhere.
)doc");
```</p>
<p>Note that:</p>
<ul>
<li>This op has three input tensors - one boolean tensor for selecting which
    values to choose from the two other input tensors of matching type <code>T</code>, and
    one output tensor of type <code>T</code>.</li>
<li>The <code>Attr</code> for this op is defined as <code>.Attr("T: type")</code> which specifies <code>T</code>
    as an <code>Attr</code> of type <code>type</code>. In the subsequent steps, you will use <code>T</code> with
    a template class to define the type of the contents of tensors.</li>
<li>The docstring for this op is specified by passing a string to <code>.Doc()</code>.</li>
<li>The shape function for this op uses the <code>Merge</code> method of the
    <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/shape_inference.h#:~:text=class%20InferenceContext"><code>tensorflow::shape_inference::InferenceContext</code></a>
    object which is a helper function to set the output shape to be the same as
    the identical shapes of the two inputs (for example, if it is used for
    binary ops) and has error checking that the two inputs have the same shape.
    Since <code>multiplex_1</code> has three inputs, two calls to <code>Merge</code> are used to
    assert that all three inputs are the same shape.</li>
</ul>
<h3>Step 2 - Register the op implementation (kernel)</h3>
<p>Register the kernel by calling the <code>REGISTER_KERNEL_BUILDER</code> macro.</p>
<p>```
#define REGISTER_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Examples1&gt;MultiplexDense")      \
                              .Device(::tensorflow::DEVICE_CPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexDenseOp<type>)</p>
<p>TF_CALL_ALL_TYPES(REGISTER_KERNELS);</p>
<h1>undef REGISTER_KERNELS</h1>
<p>```</p>
<h3>Step 3 - Implement the op kernel(s)</h3>
<p>In the op kernel in <code>multiplex_1_kernel.cc</code>, create a class derived from
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#:~:text=class%20OpKernel"><code>OpKernel</code></a>
that implements a <code>Compute</code> method to get and validate input tensors, perform
computation, and create the output tensors.</p>
<p>```
template <typename T>
class MultiplexDenseOp : public OpKernel {
 public:
  explicit MultiplexDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  MultiplexDenseOp(const MultiplexDenseOp&amp; other) = delete;
  MultiplexDenseOp&amp; operator=(const MultiplexDenseOp&amp; other) = delete;
  ~MultiplexDenseOp() override = default;</p>
<p>void Compute(OpKernelContext* ctx) override {
    const auto&amp; cond_tensor = ctx-&gt;input(0);
    const auto&amp; a_values_tensor = ctx-&gt;input(1);
    const auto&amp; b_values_tensor = ctx-&gt;input(2);</p>
<pre><code>// Allow any shape, but require that a_values, b_values, and cond all
// have the same shape.
// Note that ::tensorflow::TensorShapeUtils has some useful functions
// for checking shapes.
OP_REQUIRES(ctx, a_values_tensor.shape() == b_values_tensor.shape(),
            InvalidArgument(
                "a_values and b_values must have the same shape. "
                "a_values shape: ",
                a_values_tensor.shape().DebugString(), " b_values shape: ",
                b_values_tensor.shape().DebugString()));
OP_REQUIRES(
    ctx, a_values_tensor.shape() == cond_tensor.shape(),
    InvalidArgument("a_values and cond must have the same shape. "
                    "a_values shape: ",
                    a_values_tensor.shape().DebugString(),
                    " cond shape: ", cond_tensor.shape().DebugString()));

const auto a_values = a_values_tensor.flat&lt;T&gt;();
const auto b_values = b_values_tensor.flat&lt;T&gt;();
const auto cond = cond_tensor.flat&lt;bool&gt;();

// Create an output tensor
Tensor* output_tensor = nullptr;
OP_REQUIRES_OK(
    ctx, ctx-&gt;allocate_output(0, a_values_tensor.shape(), &amp;output_tensor));
auto output = output_tensor-&gt;template flat&lt;T&gt;();
const int64_t N = a_values_tensor.NumElements();

// Here is an example of processing tensors in a simple loop directly
// without relying on any libraries. For intensive math operations, it is
// a good practice to use libraries such as Eigen that support
// tensors when possible, e.g. "output = cond.select(a_values, b_values);"
// Eigen supports chunking into blocks and multi-threading.
// See
// https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title55
for (int64_t i = 0; i &lt; N; i++) {
  if (cond(i)) {
    output(i) = a_values(i);
  } else {
    output(i) = b_values(i);
  }
}
</code></pre>
<p>}
};
```</p>
<p>A common way to access the values in tensors for manipulation is to get
flattened rank-1
<a href="https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html"><code>Eigen::Tensor</code></a>
objects. In the example code, this is done for all three inputs and the output.
The example also processes tensors in a simple loop directly without relying on
any libraries.</p>
<p>Using Eigen, the <code>for</code> loop above could have been written simply as:</p>
<!-- test_snippets_in_readme skip -->
<p><code>c++
output = cond.select(a_values, b_values);</code></p>
<p><a href="https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title55">Selection</a>
from Eigen supports chunking into blocks and multi-threading.</p>
<p>For intensive mathematical operations, it is a good practice to use libraries
such as <a href="https://eigen.tuxfamily.org/index.php?title=Main_Page#Overview">Eigen</a>
that support tensors to do the computation when possible. Eigen is vectorized,
avoids dynamic memory allocation and therefore is typically faster than using
simple <code>for</code> loops.</p>
<p>Eigen provides the following for accessing tensor values (for both inputs and
outputs):</p>
<ul>
<li><code>flat&lt;T&gt;()(index)</code> for element access for tensors of any rank</li>
<li><code>scalar&lt;T&gt;()()</code> for rank 0 tensors</li>
<li><code>vec&lt;T&gt;()(index)</code> for rank 1 tensors</li>
<li><code>matrix&lt;T&gt;()(i, j)</code> for rank 2 tensors</li>
<li><code>tensor&lt;T, 3&gt;()(i, j, k)</code> for tensors of known rank (e.g. 3).</li>
</ul>
<h4>Compile the op (optional)</h4>
<p>Compile the C++ op to create a kernel library and Python wrapper that enables
you to use the op with TensorFlow.</p>
<p>Create a <code>BUILD</code> file for the op which declares the dependencies and the output
build targets. Refer to
<a href="https://www.tensorflow.org/guide/create_op#build_the_op_library">building for OSS</a>.</p>
<h3>Step 4 - Create the Python wrapper (optional)</h3>
<p>To create the Python wrapper, import and implement a function that serves as the
op's public API and provides a docstring.</p>
<p><code>``
def multiplex(cond, a, b, name=None):
  """Return elements chosen from</code>a<code>or</code>b<code>depending on</code>cond`.</p>
<p>This is similar to <code>np.where</code> and <code>tf.where</code>, but simplified to only handle
  the case of dense tensors, no optional parameters, no broadcasting, etc..</p>
<blockquote>
<blockquote>
<blockquote>
<p>multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])
  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  1, 200, 300,   4], ...)></p>
</blockquote>
</blockquote>
</blockquote>
<p>Args:
    cond: tf.Tensor of type bool. Where True, yield <code>a</code>, otherwise yield <code>b</code>.
    a: tf.Tensor with the same type and shape as <code>b</code>.
    b: tf.Tensor with the same type and shape as <code>a</code>.
    name: An optional name for the op.</p>
<p>Returns:
    A tf.Tensor with elements from <code>a</code> where <code>cond</code> is True, and elements
    from <code>b</code> elsewhere.
  """
  return gen_multiplex_1_op.examples1_multiplex_dense(
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
error types and/or messages.</p>
<p>Below are test excerpts showing how to handle errors for different scenarios.
The first test case demonstrates error handling when errors are common across
eager and graph executions and the second test case demonstrates error handling
when the errors are different in eager and graph executions.</p>
<p><code>@test_util.run_in_graph_and_eager_modes
  def test_multiplex_int(self):
    a = tf.constant([1, 2, 3, 4, 5])
    b = tf.constant([10, 20, 30, 40, 50])
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [1, 20, 3, 40, 5]
    result = multiplex_1_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)</code></p>
<p><code>@test_util.run_in_graph_and_eager_modes
  def test_multiplex_bad_types(self):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])  # float
    b = tf.constant([10, 20, 30, 40, 50])  # int32
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, TypeError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(cannot compute Examples1&gt;MultiplexDense as input #2\(zero-based\) '
        r'was expected to be a float tensor but is a int32 tensor '
        r'\[Op:Examples1&gt;MultiplexDense\]'
        r')|('
        # Graph mode raises TypeError with the following message
        r"Input 'b_values' of 'Examples1&gt;MultiplexDense' Op has type int32 that "
        r"does not match type float32 of argument 'a_values'.)"):
      self.evaluate(multiplex_1_op.multiplex(cond, a, b))</code></p>
<p>Refer to <code>multiplex_1_test.py</code> for the full source code which contains all the
test cases.</p>
<p>Reuse the <code>BUILD</code> file created in Step 3a above to add build
rules for the Python API wrapper and the op test.</p>
<!-- test_snippets_in_readme skip -->
<p><code>load("//third_party/tensorflow:strict.default.bzl", "py_strict_library")
load("//third_party/tensorflow:tensorflow.default.bzl", "tf_py_test")</code></p>
<p>```
py_strict_library(
    name = "multiplex_1_op",
    srcs = ["multiplex_1_op.py"],
    srcs_version = "PY3",
    visibility = ["//third_party/tensorflow/google/g3doc:<strong>subpackages</strong>"],
    deps = [
        ":gen_multiplex_1_op",
        ":multiplex_1_kernel",
        "//third_party/py/tensorflow",
    ],
)</p>
<p>tf_py_test(
    name = "multiplex_1_test",
    size = "small",
    srcs = ["multiplex_1_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":multiplex_1_op",
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
$ bazel test //third_party/tensorflow/google/g3doc/example/multiplex_1:multiplex_1_test</code></p>
<h3>Use the op</h3>
<p>Use the op by importing and calling it as follows:</p>
<!-- test_snippets_in_readme skip -->
<p>```python
import tensorflow as tf
from tensorflow.examples.custom_ops_doc.multiplex_1 import multiplex_1_op</p>
<p>a = tf.constant([1, 2, 3, 4, 5])
b = tf.constant([10, 20, 30, 40, 50])
cond = tf.constant([True, False, True, False, True], dtype=bool)</p>
<p>result = multiplex_1_op.multiplex(cond, a, b)
result.numpy()
```</p>
<!-- test_snippets_in_readme skip -->
<p><code>array([ 1, 20,  3, 40,  5], dtype=int32)</code></p>
<p>Here, <code>multiplex_1_op</code> is the name of the Python wrapper that was created in
this example.</p>
<h3>Summary</h3>
<p>In this example, you learned how to define and use a custom multiplexer op. The
image below summarizes the files created for this op.</p>
<p>The table below summarizes the build rules and targets for building and testing
the <code>multiplex_1</code> op.</p>
<p>Op components                           | Build rule             | Build target         | Source
--------------------------------------- | ---------------------- | -------------------- | ------
Kernels (C++)                           | <code>tf_custom_op_library</code> | <code>multiplex_1_kernel</code> | <code>multiplex_1_kernel.cc</code>, <code>multiplex_1_op.cc</code>
Wrapper (automatically generated)       | N/A                    | <code>gen_multiplex_1_op</code> | N/A
Wrapper (with public API and docstring) | <code>py_strict_library</code>    | <code>multiplex_1_op</code>     | <code>multiplex_1_op.py</code>
Tests                                   | <code>tf_py_test</code>           | <code>multiplex_1_test</code>   | <code>multiplex_1_test.py</code></p>
<!-- LINT.ThenChange(multiplex_1.md) -->