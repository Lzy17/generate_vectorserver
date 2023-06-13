<!-- LINT.IfChange -->
<h1>Create a custom multiplexer op with GPU support</h1>
<p>This guide provides an end-to-end example for adding a custom multiplexer op
with both CPU and GPU support.</p>
<p>For a simpler example of a TensorFlow multiplexer custom op, refer to
<code>multiplex_1</code>. The <code>multiplex_2</code> operation builds on the <code>multiplex_1</code> operation
in the following ways:</p>
<ul>
<li>This op includes support for both GPU and CPU, while <code>multiplex_1</code> only
    supports CPU.</li>
<li>This op uses the
    <a href="https://eigen.tuxfamily.org/index.php?title=Main_Page#Overview">Eigen</a>
    library to access tensor values and compute the multiplex operation. The
    <code>multiplex_1</code> op only uses Eigen to access tensor values.</li>
</ul>
<p>This example uses <code>multiplex_2_kernel.cc</code> to register the op for CPU, and
<code>multiplex_2_kernel.cu.cc</code> to register the op for GPU. Excluding the
<code>multiplex_2_kernel.cu.cc</code> file from this op will result in a multiplexer similar
to multiplex_1.</p>
<p>The content on this page assumes familiarity with the high-level process for
adding custom ops to TensorFlow. For additional context, read the
<a href="https://www.tensorflow.org/guide/create_op">OSS guide on creating custom ops</a>.</p>
<h2>Creating a custom multiplexer op with GPU support</h2>
<p>This example demonstrates how you can create a Python custom multiplexer,
<code>multiplex_2_op</code>, similar to
<a href="https://tensorflow.org/api_docs/python/tf/where?version=nightly"><code>tf.where</code></a>.
It returns elements chosen from either of the two input tensors (<code>x</code> or <code>y</code>),
depending on the <code>condition</code>. You can call the op with the following:</p>
<!-- test_snippets_in_readme skip -->
<p><code>python
multiplex_2_op.multiplex(condition, x, y)</code></p>
<p>This simplified <code>multiplex_2</code> op has the following limitations that are not
present in <code>tf.where</code>:</p>
<ul>
<li>Support only for dense tensors</li>
<li>No broadcasting capabilities</li>
<li>No extensibility through optional parameters</li>
</ul>
<p>This example contains C++ and Python code snippets to illustrate the code flow.
These snippets may be missing namespace declarations, imports, and test cases.</p>
<h3>Step 1 - Define the op interface</h3>
<p>Define the op interface and register it using the <code>REGISTER_OP</code> macro.</p>
<p>```
REGISTER_OP("Examples&gt;MultiplexDense")
    .Input("cond: bool")
    .Input("a: T")
    .Input("b: T")
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
the case of dense tensors, no optional parameters, no broadcasting, etc..
This uses cond.select from the Eigen library and supports GPU (and CPU).</p>
<p>cond: tf.Tensor of type bool.
a: tf.Tensor with the same type and shape as <code>b</code>.
b: tf.Tensor with the same type and shape as <code>a</code>.</p>
<pre><code>  Where True, yield `a`, otherwise yield `b`.
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
    binary ops) and has error checking to ensure that the two inputs have the
    same shape. Since <code>multiplex_2</code> has three inputs, two calls to <code>Merge</code> are
    used to assert that all three inputs are the same shape.</li>
</ul>
<h3>Step 2 - Register the op implementation (kernel)</h3>
<p>This example registers the kernel for both CPU and GPU. You can register the
kernel for only CPU using <code>multiplex_2_kernel.cc</code>.
This will result in a kernel similar to the <code>multiplex_1</code> custom op.
The types supported by GPU kernels are a subset of the types supported by CPU
kernels.</p>
<p>Register the kernel by calling the <code>REGISTER_KERNEL_BUILDER</code> macro.</p>
<p>```
#define REGISTER_KERNELS_GPU(type)                              \
  REGISTER_KERNEL_BUILDER(Name("Examples&gt;MultiplexDense")       \
                              .Device(::tensorflow::DEVICE_GPU) \
                              .TypeConstraint<type>("T"),       \
                          MultiplexDenseOp<GPUDevice, type>)</p>
<p>REGISTER_KERNELS_GPU(bool);
REGISTER_KERNELS_GPU(Eigen::half);
REGISTER_KERNELS_GPU(float);
REGISTER_KERNELS_GPU(double);
REGISTER_KERNELS_GPU(int64);
REGISTER_KERNELS_GPU(complex64);
REGISTER_KERNELS_GPU(complex128);</p>
<h1>undef REGISTER_KERNELS_GPU</h1>
<p>```</p>
<h3>Step 3 - Implement the op kernel(s)</h3>
<p>In the op kernel (<code>multiplex_2_kernel.h</code>), create a class derived from
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#:~:text=class%20OpKernel"><code>OpKernel</code></a>
that implements a <code>Compute</code> method to get and validate input tensors, perform
computation, and create the output tensors. This file is included by both
<code>multiplex_2_kernel.cu.cc</code> (for GPU) and <code>multiplex_2_kernel.cc</code> (for CPU).</p>
<p>```
template <typename Device, typename T>
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
            ::tensorflow::errors::InvalidArgument(
                "a and b must have the same shape. "
                "a shape: ",
                a_values_tensor.shape().DebugString(),
                " b shape: ", b_values_tensor.shape().DebugString()));
OP_REQUIRES(ctx, a_values_tensor.shape() == cond_tensor.shape(),
            ::tensorflow::errors::InvalidArgument(
                "a and cond must have the same shape. "
                "a shape: ",
                a_values_tensor.shape().DebugString(),
                " cond shape: ", cond_tensor.shape().DebugString()));
OP_REQUIRES(ctx, a_values_tensor.NumElements() &gt; 0,
            ::tensorflow::errors::InvalidArgument(
                "Inputs must have at least one element."));

const auto a_values = a_values_tensor.flat&lt;T&gt;();
const auto b_values = b_values_tensor.flat&lt;T&gt;();
const auto cond = cond_tensor.flat&lt;bool&gt;();

// Create an output tensor
Tensor* output_tensor = nullptr;
OP_REQUIRES_OK(
    ctx, ctx-&gt;allocate_output(0, a_values_tensor.shape(), &amp;output_tensor));
auto output = output_tensor-&gt;template flat&lt;T&gt;();
// Here is an example of processing tensors using the Eigen library.
// This supports both CPU and GPU.
// For CPU, it supports chunking into blocks and multi-threading.
// See
// https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title55
output.device(ctx-&gt;eigen_device&lt;Device&gt;()) =
    cond.select(a_values, b_values);
</code></pre>
<p>}
};
```</p>
<p>For intensive mathematical operations, it is a good practice to use
<a href="https://eigen.tuxfamily.org/index.php?title=Main_Page#Overview">Eigen</a> to
perform the computation. Eigen is vectorized, avoids dynamic memory allocation
and is faster on tensors.The definitions related to Eigen are:</p>
<!-- test_snippets_in_readme skip -->
<p>```c++</p>
<h1>define EIGEN_USE_THREADS</h1>
<h1>if GOOGLE_CUDA || TENSORFLOW_USE_ROCM</h1>
<h1>define EIGEN_USE_GPU</h1>
<h1>endif</h1>
<p>```</p>
<p><a href="https://eigen.tuxfamily.org/dox/unsupported/eigen_tensors.html#title55">Selection</a>
from Eigen supports CPU and GPU devices, as well as chunking data into blocks
and multi-threading. The <code>multiplex_2</code> op contains the following:</p>
<!-- test_snippets_in_readme skip -->
<p><code>c++
output.device(ctx-&gt;eigen_device&lt;Device&gt;()) =
     cond.select(a_values, b_values);</code></p>
<p>Using Eigen simplified this example. Alternatively, Custom Ops may implement
kernels for GPU directly in the <code>*.cu.cc</code> files using C++.</p>
<h4>Compile the op</h4>
<p>Compile the C++ op to create a kernel library and Python wrapper that enables
you to use the op with TensorFlow.</p>
<p>Create a <code>BUILD</code> file for the op which declares the dependencies and the output
build targets. Refer to
<a href="https://www.tensorflow.org/guide/create_op#build_the_op_library">building for OSS</a>.</p>
<h3>Step 4 - Create the Python wrapper</h3>
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
  return gen_multiplex_2_op.examples_multiplex_dense(
      cond=cond, a=a, b=b, name=name)
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
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [1, 20, 3, 40, 5]
    result = multiplex_2_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)</code></p>
<p><code>@test_util.run_in_graph_and_eager_modes
  def test_multiplex_bad_types(self):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])  # float
    b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, TypeError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(cannot compute Examples&gt;MultiplexDense as input #2\(zero-based\) '
        r'was expected to be a float tensor but is a int64 tensor '
        r'\[Op:Examples&gt;MultiplexDense\]'
        r')|('
        # Graph mode raises TypeError with the following message
        r"Input 'b' of 'Examples&gt;MultiplexDense' Op has type int64 that "
        r"does not match type float32 of argument 'a'.)"):
      self.evaluate(multiplex_2_op.multiplex(cond, a, b))</code></p>
<p>Refer to <code>multiplex_2_test.py</code> for the full source code which contains all the
test cases.</p>
<p>Reuse the <code>BUILD</code> file to add build rules for the Python API wrapper and the op
test.</p>
<p>```
py_strict_library(
    name = "multiplex_2_op",
    srcs = ["multiplex_2_op.py"],
    data = ["multiplex_2_kernel.so"],
    srcs_version = "PY3",
    visibility = ["//third_party/tensorflow/examples/custom_ops_doc:<strong>subpackages</strong>"],
    deps = [
        "//third_party/py/tensorflow",
    ],
)</p>
<p>cuda_py_test(
    name = "multiplex_2_test",
    size = "small",
    srcs = ["multiplex_2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
        "no_pip",
    ],
    deps = [
        ":multiplex_2_op",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python/framework:errors",
        "//third_party/tensorflow/python/framework:test_lib",
    ],
)
```</p>
<p>Test the op in the following ways:</p>
<ul>
<li>
<p>Build for CPU and test on CPU</p>
<p><!-- test_snippets_in_readme skip -->
<code>shell
bazel test //third_party/tensorflow/google/g3doc/example/multiplex_2:multiplex_2_test</code></p>
</li>
<li>
<p>Build for GPU and CPU; test on CPU</p>
<p><!-- test_snippets_in_readme skip -->
<code>shell
$ bazel test --config=cuda //third_party/tensorflow/google/g3doc/example/multiplex_2:multiplex_2_test</code></p>
</li>
<li>
<p>Build for GPU and CPU; test on GPU (note the <code>_gpu</code> suffix in the target)</p>
<p><!-- test_snippets_in_readme skip -->
<code>shell
$ bazel test --config=cuda //third_party/tensorflow/google/g3doc/example/multiplex_2:multiplex_2_test_gpu</code></p>
</li>
</ul>
<p>Testing and building exclusively on CPU only requires the multiplex_2_kernel.cc
file when registering the op. For all other cases, include both
multiplex_2_kernel.cc and multiplex_2_kernel.cu.cc files.</p>
<h3>Use the op</h3>
<p>Import the op and call it using the following example:</p>
<!-- test_snippets_in_readme skip -->
<p>```python
import tensorflow as tf</p>
<p>from tensorflow.examples.custom_ops_doc.multiplex_2 import multiplex_2_op</p>
<p>a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
cond = tf.constant([True, False, True, False, True], dtype=bool)</p>
<h1>expected result is [1, 20, 3, 40, 5]</h1>
<p>result = multiplex_2_op.multiplex(cond, a, b)
```</p>
<p>Here, <code>multiplex_2_op</code> is the name of the Python wrapper that was created in
this example.</p>
<p>When running an op on GPU, use inputs with types supported by the GPU kernels
(e.g. this example uses <code>tf.int64</code> for <code>a</code> and <code>b</code> since this type was
registered).</p>
<h3>Summary</h3>
<p>In this example, you learned how to define and use a custom multiplexer op for
GPU. The image below summarizes the files created for this op.</p>
<p>The table below summarizes the build rules and targets for building and testing
the <code>multiplex_2</code> op.</p>
<p>Op components                           | Build rule             | Build target         | Source
--------------------------------------- | ---------------------- | -------------------- | ------
Kernels (C++)                           | <code>tf_custom_op_library</code> | <code>multiplex_2_kernel</code> | <code>multiplex_2_kernel.cu.cc</code>, <code>multiplex_2_kernel.cc</code>, <code>multiplex_2_op.cc</code>, <code>multiplex_2_kernel.h</code>
Wrapper (automatically generated)       | N/A                    | <code>gen_multiplex_2_op</code> | N/A
Wrapper (with public API and docstring) | <code>py_strict_library</code>    | <code>multiplex_2_op</code>     | <code>multiplex_2_op.py</code>
Tests                                   | <code>cuda_py_test</code>         | <code>multiplex_2_test</code>   | <code>multiplex_2_test.py</code></p>
<!-- LINT.ThenChange(multiplex_2.md) -->