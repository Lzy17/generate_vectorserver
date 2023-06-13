<p>This folder contains a convenience library called <em>tf-shim</em> over TF and TFLite
op kernel APIs.</p>
<h2>Summary</h2>
<p>This library creates a shim over the custom op APIs of TF and TFLite so the
developer can write the custom op once with minimal binary or runtime overhead.</p>
<p>An example usage is an input preprocessing op kernel that can be used in
both TF and TFLite.</p>
<h2>Background</h2>
<p>When there is a need to implement a logic that is not supported by the TF
builtin ops the alternative is to build a custom op. If that op needs to
run on-device then it needs to be written in C++ against the client API for
custom ops.</p>
<p>For example, feature processing especially for textual input in an ML model
can involve operations that don't lend themselves well to vectorization and the
code, if written as a C++ function, would be much shorter and more readable.</p>
<p>However, Tensorflow and TFLite APIs for creating op kernels are, at the moment,
not identical. This library offers a convenient way to write the kernel once and
adapt it to both TF and TFLite with minimal binary and runtime overhead.</p>
<h2>Implementation</h2>
<p>This folder contains two pieces:</p>
<ol>
<li>
<p><code>TensorView</code> as a shim over <code>::tensorflow::Tensor</code> and <code>TfLiteTensor</code></p>
</li>
<li>
<p><code>OpKernelShim</code> class which abstracts the TF and TFLite op kernel APIs.</p>
</li>
</ol>
<h3>TensorView</h3>
<p>This class is a <em>view</em> over an already allocated tensor in TF or TFLite without
taking any ownership. In that sense it is similar to <code>std::string_view</code> but with
the difference that the underlying buffer can be mutable.</p>
<p>Example Usage:</p>
<p>```
::tensorflow::Tensor tf_tensor;
auto t = TensorView::New(&amp;tf_tensor);</p>
<p>auto t_str_mat = t.As&lt;::tensorflow::tstring, /<em>RANK=</em>/ 2&gt;();
t(0, 0) = "ab";
t(0, 1) = "cde"</p>
<p>auto t_buffer = t.Data&lt;::tensorflow::tstring&gt;();
t[0] = "ab";
t[1] = "cde"
```</p>
<p>```
TfLiteTensor tflite_tensor;
auto t = TensorView::New(&amp;tflite_tensor);</p>
<p>auto t_int_vec = t.As<int32, /*RANK=*/ 1>();
t(0) = 123;
t(1) = 456</p>
<p>auto t_buffer = t.Data<int32>();
t[0] = 123;
t[1] = 456
```</p>
<p>The <code>New</code> is the factory function which based on the type of the input returns
either a <code>TfTensorView</code> or a <code>TfLiteTensorView</code>.</p>
<p>See the unit tests <code>tf_tensor_view_test.cc</code> and <code>tflite_tensor_view_test.cc</code> for
more usage.</p>
<p>The string tensor in <code>TfLiteTensorView</code> is a bit of special case. Since string
tensors in TfLite are serialized in a specific format, while writing to those
tensors an intermediate buffer is needed to hold intermediate values before all
the strings get serialized. The intermediate string buffers are serialized back
to the TfLite string format once the last remaining <code>TfLiteTensorView</code> goes out
of scope. Only then the user can see the string values in the underlying
<code>TfLiteTensor</code>. That said, when implementing an op kernel, there is rarely a
need to read back the contents of a mutable output <code>TfLiteTensor</code> within the
same code block.</p>
<h3>OpKernelShim</h3>
<p><em>WARNING: Experimental interface, subject to change</em></p>
<p>This class defines the interface which when implemented allows for convenient
adaptation to TF and TFLite op kernels.</p>
<p>Here is an example op kernel implementing this interface:</p>
<p>```
template<TfRuntime R>
class MyOp : public OpKernelShim<MyOp, R> {</p>
<p>// Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs();</p>
<p>// Input tensors declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs();</p>
<p>// Output tensors declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs();</p>
<p>// Initializes the op
  absl::Status Init(InitContext* ctx);</p>
<p>// Runs the operation
  absl::Status Invoke(InvokeContext* ctx);</p>
<p>// Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* ctx);
};
```</p>
<p>The class <code>MyOp</code> is passing itself to <code>OpKernelShim</code> as a template parameter.
This is because <code>OpKernelShim</code> is a static interface using the CRTP pattern.
Similarly, the context classes: <code>InitContext</code>, <code>InvokeContext</code> and
<code>ShapeInferenceContext</code> are all static interfaces in the same way.</p>
<p>The class <code>MyOp</code> can also be templatized. See <code>test_op/tmpl_op.h</code> for an
example.</p>
<h3>Context Interfaces</h3>
<p>An op kernel written using this library has access to a number of <em>context</em>
objects at various stages of its lifecycle. These context objects are
effectively shims over the existing context objects in TF and TFLite.</p>
<h4>InitContext</h4>
<p>An instance of this class is passed to the op kernel during its initialization.</p>
<p><code>template &lt;typename SubType&gt;
class InitContext {
 public:
  // Read the given attribute and populate the given value.
  template &lt;typename AttrType&gt;
  absl::Status GetAttr(const std::string&amp; attr_name, AttrType* value) const;
};</code></p>
<h4>InvokeContext</h4>
<p>An instance of this class is passed to the op kernel during its invocation.</p>
<p><code>template &lt;typename SubType&gt;
class InvokeContext {
 public:
  // Read an input tensor
  ConstTensorViewOr GetInput(const int idx) const;
  // Get a mutable output tensor
  TensorViewOr GetOutput(const int idx, const Shape&amp; shape) const;
};</code></p>
<h4>ShapeInferenceContext</h4>
<p>An instance of this class is passed to the op kernel during its shape inference.</p>
<p><code>template &lt;typename SubType&gt;
class ShapeInferenceContext {
 public:
  // Read an input tensor shape
  ShapeOr GetInputShape(const int idx) const;
  // Set an output tensor shape
  absl::Status SetOutputShape(const int idx, const Shape&amp; shape);
  // Read an input tensor during shape inference
  ConstTensorViewOr GetInputTensor(const int idx) const;
};</code></p>