<h1>TFLite on GPU</h1>
<p>TensorFlow Lite (TFLite) supports several hardware accelerators.  This document
describes how to use the GPU backend using the TFLite delegate APIs on Android
and iOS.</p>
<p>GPUs are designed to have high throughput for massively parallelizable
workloads.  Thus, they are well-suited for deep neural nets which consists of a
huge number of operators, each working on some input tensor(s) that can be
easily divided into smaller workloads and carried out in parallel, typically
resulting in lower latency.  In the best scenario, inference on the GPU may now
run fast enough and now become suitable for real-time applications if it was not
before.</p>
<p>GPUs do their computation with 16-bit or 32-bit floating point numbers and do
not require quantization for optimal performance unlike the CPUs.  If
quantization of your neural network was not an option due to lower accuracy
caused by lost precision, such concern can be discarded when running deep neural
net models on the GPU.</p>
<p>Another benefit that comes with GPU inference is its power efficiency.  GPUs
carry out the computations in a very efficient and optimized way, so that they
consume less power and generate less heat than when the same task is run on the
CPUs.</p>
<p>TFLite on GPU supports the following ops in 16-bit and 32-bit float precision:</p>
<ul>
<li><code>ADD v1</code></li>
<li><code>AVERAGE_POOL_2D v1</code></li>
<li><code>CONCATENATION v1</code></li>
<li><code>CONV_2D v1</code></li>
<li><code>DEPTHWISE_CONV_2D v1-2</code></li>
<li><code>EXP v1</code></li>
<li><code>FULLY_CONNECTED v1</code></li>
<li><code>LOGISTIC v1</code></li>
<li><code>LSTM v2 (Basic LSTM only)</code></li>
<li><code>MAX_POOL_2D v1</code></li>
<li><code>MAXIMUM v1</code></li>
<li><code>MINIMUM v1</code></li>
<li><code>MUL v1</code></li>
<li><code>PAD v1</code></li>
<li><code>PRELU v1</code></li>
<li><code>RELU v1</code></li>
<li><code>RELU6 v1</code></li>
<li><code>RESHAPE v1</code></li>
<li><code>RESIZE_BILINEAR v1-3</code></li>
<li><code>SOFTMAX v1</code></li>
<li><code>STRIDED_SLICE v1</code></li>
<li><code>SUB v1</code></li>
<li><code>TRANSPOSE_CONV v1</code></li>
</ul>
<h2>Basic Usage</h2>
<p><strong>Note:</strong> Following section describes the example usage for Android GPU delegate
with C++. For other languages and platforms, please see
<a href="https://www.tensorflow.org/lite/performance/gpu">the documentation</a>.</p>
<p>Using TFLite on GPU is as simple as getting the GPU delegate via
<code>TfLiteGpuDelegateV2Create()</code> and then passing it to
<code>InterpreterBuilder::AddDelegate()</code>:</p>
<p>```c++
////////
// Set up InterpreterBuilder.
auto model = FlatBufferModel::BuildFromFile(model_path);
ops::builtin::BuiltinOpResolver op_resolver;
InterpreterBuilder interpreter_builder(*model, op_resolver);</p>
<p>////////
// NEW: Prepare GPU delegate.
auto<em> delegate = TfLiteGpuDelegateV2Create(/</em>default options=*/nullptr);
interpreter_builder.AddDelegate(delegate);</p>
<p>////////
// Set up Interpreter.
std::unique_ptr<Interpreter> interpreter;
if (interpreter_builder(&amp;interpreter) != kTfLiteOk) return;</p>
<p>////////
// Run inference.
WriteToInputTensor(interpreter-&gt;typed_input_tensor<float>(0));
if (interpreter-&gt;Invoke() != kTfLiteOk) return;
ReadFromOutputTensor(interpreter-&gt;typed_output_tensor<float>(0));</p>
<p>////////
// Clean up.
TfLiteGpuDelegateV2Delete(delegate);
```</p>
<p><em>IMPORTANT:</em> When calling <code>Interpreter::ModifyGraphWithDelegate()</code> or
<code>InterpreterBuilder::operator()</code> or
<code>Interpreter::Invoke()</code>, the caller must have a <code>EGLContext</code> in the current
thread and <code>Interpreter::Invoke()</code> must be called from the same <code>EGLContext</code>.
If such <code>EGLContext</code> does not exist, the delegate will internally create one,
but then the developer must ensure that <code>Interpreter::Invoke()</code> is always called
from the same thread <code>InterpreterBuilder::operator()</code> or
<code>Interpreter::ModifyGraphWithDelegate()</code> was called.</p>
<h2>Building and Runtime</h2>
<p>TFLite GPU backend uses OpenGL ES 3.1 compute shaders or OpenCL.</p>
<p><code>sh
bazel build --config android_arm64 //path/to/your:project</code></p>
<p>Metal shaders are used for iOS, which were introduced with iOS 8.  Thus,
compilation flags should look like:</p>
<p><code>sh
bazel build --config ios_fat //path/to/your:project</code></p>
<h2>Advanced Usage: Delegate Options</h2>
<p>There are GPU options that can be set and passed on to
<code>TfLiteGpuDelegateV2Create()</code>. When option is set to <code>nullptr</code> as shown in the
Basic Usage, it translates to:</p>
<p><code>c++
const TfLiteGpuDelegateOptionsV2 kDefaultOptions =
    TfLiteGpuDelegateOptionsV2Default();</code></p>
<p>Similar for <code>TFLGpuDelegateCreate()</code>:</p>
<p><code>c++
const TFLGpuDelegateOptions kDefaultOptions = {
  .allow_precision_loss = false,
  .wait_type = TFLGpuDelegateWaitTypePassive,
  .enable_quantization = false,
};</code></p>
<p>While it is convenient to just supply <code>nullptr</code>, it is recommended to explicitly
set the options to avoid any unexpected artifacts in case default values are
changed.</p>
<p><em>IMPORTANT:</em> Note that the default option may not be the fastest. For faster
execution, you may want to set <code>allow_precision_loss</code> to <code>true</code> so that the GPU
performs FP16 calculation internally, and set <code>wait_type</code> to
<code>TFLGpuDelegateWaitTypeAggressive</code> to avoid GPU sleep mode.</p>
<h2>Tips and Tricks</h2>
<ul>
<li>
<p>Some operations that are trivial on CPU side may be high cost in GPU land.
  One class of such operation is various forms of reshape operations (including
  <code>BATCH_TO_SPACE</code>, <code>SPACE_TO_BATCH</code>, <code>SPACE_TO_DEPTH</code>, etc.).  If those ops
  are inserted into the network just for the network architect's logical
  thinking, it is worth removing them for performance.</p>
</li>
<li>
<p>On GPU, tensor data is sliced into 4-channels.  Thus, a computation on a
  tensor of shape <code>[B, H, W, 5]</code> will perform about the same on a tensor of
  shape <code>[B, H, W, 8]</code>, but significantly worse than <code>[B, H, W, 4]</code>.</p>
</li>
<li>
<p>In that sense, if the camera hardware supports image frames in RGBA, feeding
  that 4-channel input is significantly faster as a memory copy (from 3-channel
  RGB to 4-channel RGBX) can be avoided.</p>
</li>
<li>
<p>For performance <a href="https://www.tensorflow.org/lite/performance/best_practices">best practices</a>, do not hesitate to re-train your classifier with
  mobile-optimized network architecture.  That is a significant part of
  optimization for on-device inference.</p>
</li>
</ul>
<h2>Publication</h2>
<ul>
<li><a href="https://arxiv.org/abs/1907.01989">On-Device Neural Net Inference with Mobile GPUs</a><ul>
<li>Juhyun Lee, Nikolay Chirkov, Ekaterina Ignasheva, Yury Pisarchyk, Mogan
    Shieh, Fabio Riccardi, Raman Sarokin, Andrei Kulik, and Matthias
    Grundmann</li>
<li>CVPR Workshop
    <a href="https://sites.google.com/corp/view/ecv2019">Efficient Deep Learning for Computer Vision (ECV2019)</a></li>
</ul>
</li>
</ul>