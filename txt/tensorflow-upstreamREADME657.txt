<h1>XNNPACK backend for TensorFlow Lite</h1>
<p>XNNPACK is a highly optimized library of neural network inference operators for
ARM, x86, and WebAssembly architectures in Android, iOS, Windows, Linux, macOS,
and Emscripten environments. This document describes how to use the XNNPACK
library as an inference engine for TensorFlow Lite.</p>
<h2>Using XNNPACK engine with TensorFlow Lite interpreter</h2>
<p>XNNPACK integrates with TensorFlow Lite interpreter through the delegation
mechanism. TensorFlow Lite supports several methods to enable XNNPACK
for floating-point inference.</p>
<h3>Enable XNNPACK via Java API on Android (recommended on Android)</h3>
<p>Pre-built <a href="https://www.tensorflow.org/lite/guide/android#use_the_tensorflow_lite_aar_from_mavencentral">nightly TensorFlow Lite binaries for Android</a>
include XNNPACK, albeit it is disabled by default. Use the <code>setUseXNNPACK</code>
method in <code>Interpreter.Options</code> class to enable it:</p>
<p><code>java
Interpreter.Options interpreterOptions = new Interpreter.Options();
interpreterOptions.setUseXNNPACK(true);
Interpreter interpreter = new Interpreter(model, interpreterOptions);</code></p>
<h3>Enable XNNPACK via Swift/Objective-C API on iOS (recommended on iOS)</h3>
<p>Pre-built <a href="https://www.tensorflow.org/lite/guide/ios#specifying_versions">nightly TensorFlow Lite CocoaPods</a>
include XNNPACK, but do not enable it by default. Swift developers can use
<code>InterpreterOptions</code> object to enable XNNPACK:</p>
<p><code>swift
var options = InterpreterOptions()
options.isXNNPackEnabled = true
var interpreter = try Interpreter(modelPath: "model/path", options: options)</code></p>
<p>Objective-C developers can enable XNNPACK via a new property in the
<code>TFLInterpreterOptions</code> class:</p>
<p><code>objc
TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
options.useXNNPACK = YES;
NSError *error;
TFLInterpreter *interpreter =
    [[TFLInterpreter alloc] initWithModelPath:@"model/path"
                                      options:options
                                        error:&amp;error];</code></p>
<h3>Enable XNNPACK via Bazel build flags (recommended on desktop)</h3>
<p>When building TensorFlow Lite with Bazel, add
<code>--define tflite_with_xnnpack=true</code>, and the TensorFlow Lite interpreter will
use XNNPACK engine by default.</p>
<p>The exact command depends on the target platform, e.g. for Android AAR you'd use</p>
<p><code>bazel build -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --define android_dexmerger_tool=d8_dexmerger \
  --define android_incremental_dexing_tool=d8_dexbuilder \
  --define tflite_with_xnnpack=true \
  //tensorflow/lite/java:tensorflow-lite</code></p>
<p>Note that in this case <code>Interpreter::SetNumThreads</code> invocation does not take
effect on number of threads used by XNNPACK engine. In order to specify number
of threads available for XNNPACK engine you should manually pass the value when
constructing the interpreter. The snippet below illustrates this assuming you
are using <code>InterpreterBuilder</code> to construct the interpreter:</p>
<p>```c++
// Load model
tflite::Model* model;
...</p>
<p>// Construct the interprepter
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;</p>
<p>TfLiteStatus res = tflite::InterpreterBuilder(model, resolver, num_threads);
```</p>
<p><strong>XNNPACK engine used by TensorFlow Lite interpreter uses a single thread for
inference by default.</strong></p>
<h3>Enable XNNPACK via additional dependency</h3>
<p>Another way to enable XNNPACK is to build and link the
<code>//tensorflow/lite:tflite_with_xnnpack</code> target into your application alongside
the TensorFlow Lite framework.</p>
<p>This method works on platforms which support POSIX-style weak symbols (Android,
iOS, Linux, Mac, but <strong>NOT</strong> Windows).</p>
<h3>Enable XNNPACK via low-level delegate API (not recommended)</h3>
<p>While it is possible to use low-level delegate API to enable XNNPACK, this
method is <strong>NOT RECOMMENDED</strong> unless you need to use TensorFlow Lite both with
and without XNNPACK (e.g. for benchmarking).</p>
<p>With low-level delegate API users create an XNNPACK delegate with the
<code>TfLiteXNNPackDelegateCreate</code> function, and then call
<code>Interpreter::ModifyGraphWithDelegate</code> to delegate supported parts of
the model to the XNNPACK delegate. The users must destroy the delegate with
<code>TfLiteXNNPackDelegateDelete</code> <strong>after</strong> releasing the TensorFlow Lite
interpreter. The snippet below illustrates the typical usage:</p>
<p>```c++
// Build the interpreter
std::unique_ptr<tflite::Interpreter> interpreter;
...</p>
<p>// IMPORTANT: initialize options with TfLiteXNNPackDelegateOptionsDefault() for
// API-compatibility with future extensions of the TfLiteXNNPackDelegateOptions
// structure.
TfLiteXNNPackDelegateOptions xnnpack_options =
    TfLiteXNNPackDelegateOptionsDefault();
xnnpack_options.num_threads = num_threads;</p>
<p>TfLiteDelegate* xnnpack_delegate =
    TfLiteXNNPackDelegateCreate(&amp;xnnpack_options);
if (interpreter-&gt;ModifyGraphWithDelegate(xnnpack_delegate) != kTfLiteOk) {
  // Report error and fall back to another delegate, or the default backend
}</p>
<p>...</p>
<p>// Run inference using XNNPACK
interpreter-&gt;Invoke()</p>
<p>...</p>
<p>// IMPORTANT: release the interpreter before destroying the delegate
interpreter.reset();
TfLiteXNNPackDelegateDelete(xnnpack_delegate);
```</p>
<h3>Using the XNNPACK weights cache</h3>
<p>XNNPACK internally packs static weights for operations (like convolutions) in
order to make accessing weights more memory friendly. XNNPACK needs to allocate
memory internally to hold these packed weights. If you are starting multiple
TFLite interpreter instances based on the same model, there can be multiple
copies of the same packed weights in each instance. This can cause high memory
usage. The weights cache can be used to share packed weights between multiple
TFLite instances.</p>
<p>```c++
// Create 2 interpreters which share the same model.
std::unique_ptr<tflite::Interpreter> interpreter1;
std::unique_ptr<tflite::Interpreter> interpreter2;</p>
<p>// Create a weights cache that you can pass to XNNPACK delegate.
TfLiteXNNPackDelegateWeightsCache* weights_cache =
    TfLiteXNNPackDelegateWeightsCacheCreate();</p>
<p>// Like using the low-level API above, initialize options, and pass this cache
// to XNNPACK delegate via the options.
TfLiteXNNPackDelegateOptions xnnpack_options =
    TfLiteXNNPackDelegateOptionsDefault();
xnnpack_options.weights_cache = weights_cache;</p>
<p>// Modify graph with delegate, as above...
TfLiteDelegate<em> delegate1 = TfLiteXNNPackDelegateCreate(&amp;xnnpack_options);
if (interpreter1-&gt;ModifyGraphWithDelegate(delegate1) != kTfLiteOk) {
    // Static weights will be packed and written into weights_cache.
}
TfLiteDelegate</em> delegate2 = TfLiteXNNPackDelegateCreate(&amp;xnnpack_options);
if (interpreter1-&gt;ModifyGraphWithDelegate(delegate2) != kTfLiteOk) {
    // XNNPACK will reuse packed weights if they can be found in the weights
    // cache.
}</p>
<p>// Finalize the weights cache.
// Hard finalization has the lowest memory overhead, but requires that all
// TFLite interpreter instances must be created up front before any finalization
// and inference.
TfLiteXNNPackDelegateWeightsCacheFinalizeHard(weights_cache);</p>
<p>// Alternatively, soft-finalizate the weights cache. This is useful if more
// delegates using the same model will to be created after finalization.
// TfLiteXNNPackDelegateWeightsCacheFinalizeSoft(weights_cache);</p>
<p>// Later, after all the interpreters and XNNPACK delegates using the cache are
// destroyed, release the weights cache.
TfLiteXNNPackDelegateWeightsCacheDelete(weights_cache);
```</p>
<p>The weights cache is a contents-based cache. Every time XNNPACK has to pack
weights, it first packs into a temporary buffer, then tries to look up if the
packed weights can be found in the weights cache, based on the contents of the
packed weights. If it can be found, we access the packed weights in the
cache for subsequent operations, and the temporary buffer is freed. Otherwise,
the packed weights is added to the cache.</p>
<p>The weights cache has to be finalized before any inference, it will be an error
otherwise. Hard finalization and soft finalization depends on whether new
XNNPACK delegate instances will be created after finalization. Hard finalization
does not allow new instances to be created, and has lower memory overhead. Soft
finalization allows new instances to be created, and has higher memory overhead
(up to the size of the largest packed weights, rounded up to page alignment).</p>
<h3>Using XNNPACK for variable operations</h3>
<p>XNNPACK can handle resource variables and associated operations: <code>VAR_HANDLE</code>,
<code>READ_VARIABLE</code>, and <code>ASSIGN_VARIABLE</code>, but needs to be opted in by the user
using delegate options:</p>
<p><code>c++
TfLiteXNNPackDelegateOptions xnnpack_options =
    TfLiteXNNPackDelegateOptionsDefault();
xnnpack_options.handle_variable_ops = true;</code></p>
<p>When XNNPACK handles resource variables,
<a href="https://github.com/tensorflow/tensorflow/blob/5b4239ba9cf127fd26cd9f03c04dfc4c94c078d4/tensorflow/lite/core/subgraph.h#L197">tflite::Subgraph::resources</a>
cannot be used to access resources, because the resources are now internal to
XNNPACK, and the changes are not reflected in tflite::Subgraph::resources. There
is currently no way to access resources if XNNPACK handles resource variables.</p>
<h2>Profiling</h2>
<p>When TfLite profiling is enabled, XNNPACK will time each operator and report the
results to TfLite which will print them as part of the overall execution profile.</p>
<h2>Limitations and supported operators</h2>
<p>XNNPACK delegate is a work-in-progress, and currently supports a limited set of
operators. Unsupported operators will fall back to the default implementations,
so models using a combination of supported and unsupported operators can still
benefit from XNNPACK delegate.</p>
<h3>Floating-Point (IEEE FP32) Operators</h3>
<p>Below is the list of currently supported floating-point operators:</p>
<h4><code>ABS</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>ADD</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Only addition with two inputs is supported.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>AVERAGE_POOL_2D</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>1x1 pooling with non-unit stride is not supported.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>CEIL</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>CONCATENATION</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Only concatenation with two, three, or four inputs is supported.</li>
</ul>
<h4><code>CONV_2D</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Bias is mandatory.</li>
<li>Both filter and bias must be static (use <code>kTfLiteMmapRo</code> allocation type).</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>DEPTH_TO_SPACE</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Block size must be greater than 1.</li>
</ul>
<h4><code>DEPTHWISE_CONV_2D</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Bias is mandatory.</li>
<li>Both filter and bias must be static (use <code>kTfLiteMmapRo</code> allocation type).</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>DIV</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>ELU</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>FULLY_CONNECTED</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Both filter and bias must be static (use <code>kTfLiteMmapRo</code> allocation type).</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>FLOOR</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>HARD_SWISH</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>LEAKY_RELU</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>LOGISTIC</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>MAX_POOL_2D</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>1x1 pooling with non-unit stride is not supported.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>MAXIMUM</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>MEAN</code></h4>
<ul>
<li>The first input and the output must be 4D tensors in 32-bit
  floating-point format.</li>
<li>The second input (the input with the axes specification) must be static
  (use <code>kTfLiteMmapRo</code> allocation type).</li>
<li>Only [1, 2], [2, 1], and [2] axes specification (i.e. reduction across either
  both spatial dimensions or across the width dimension) is supported.</li>
</ul>
<h4><code>MINIMUM</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>MUL</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>NEG</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>PAD</code></h4>
<ul>
<li>The first input and the output must be in 32-bit floating-point format.</li>
<li>The second input (the input with the padding specification) must be static
  (use <code>kTfLiteMmapRo</code> allocation type).</li>
<li>The numbers of padding elements must be non-negative.</li>
</ul>
<h4><code>PRELU</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Slope must be static (use <code>kTfLiteMmapRo</code> allocation type).</li>
<li>Slope must be either a 1D tensor, or have all its non-channel dimensions equal
  1.</li>
</ul>
<h4><code>RELU</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>RELU6</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>RELU_N1_TO_1</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>RESHAPE</code></h4>
<ul>
<li>The first input and the output must be in 32-bit floating-point format.</li>
<li>The second input (the input with the new shape specification) must be either
  static (use <code>kTfLiteMmapRo</code> allocation type), or absent (with the new shape
  specified via <code>ReshapeOptions</code> table).</li>
</ul>
<h4><code>RESIZE_BILINEAR</code></h4>
<ul>
<li>The first input and the output must be 4D tensors in 32-bit floating-point
  format.</li>
<li>The second input (the input with the new shape specification) must be
  static (use <code>kTfLiteMmapRo</code> allocation type).</li>
</ul>
<h4><code>ROUND</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>SLICE</code></h4>
<ul>
<li>The first input and the output must be in 32-bit floating-point format.</li>
<li>The second and third inputs (the inputs with the slices' begin and size
  specification) must be static (use <code>kTfLiteMmapRo</code> allocation type).</li>
</ul>
<h4><code>SOFTMAX</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Only <code>beta = 1.0</code> is supported.</li>
</ul>
<h4><code>SPACE_TO_DEPTH</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Block size must be greater than 1.</li>
</ul>
<h4><code>SPLIT</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Only split into two, three, or four outputs is supported.</li>
</ul>
<h4><code>SQRT</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>SQUARE</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>SQUARED_DIFFERENCE</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>STRIDED_SLICE</code></h4>
<ul>
<li>The first input and the output must be in 32-bit floating-point format.</li>
<li>The second, third, and fourth inputs (the inputs with the slices' begin, end,
  and stride specification) must be static (use <code>kTfLiteMmapRo</code> allocation
  type).</li>
<li>The fourth input (strides) must be all ones.</li>
<li>The ellipsis mask, new axis mask, and shrink axis mask must be 0.</li>
</ul>
<h4><code>SUB</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>TANH</code></h4>
<ul>
<li>Inputs and outputs must be in 32-bit floating-point format.</li>
</ul>
<h4><code>TRANSPOSE</code></h4>
<ul>
<li>The first input and the output must be in 32-bit floating-point format.</li>
<li>The second input (the input with the permutation specification) must be
  static (use <code>kTfLiteMmapRo</code> allocation type).</li>
</ul>
<h4><code>TRANSPOSE_CONV</code></h4>
<ul>
<li>Input, filter, bias (if present) and output tensors must be in 32-bit
  floating-point format.</li>
<li>Output size, filter and bias (if present) must be static (use
  <code>kTfLiteMmapRo</code> allocation type).</li>
</ul>
<h3>Floating-Point (IEEE FP16) Operators</h3>
<p>XNNPACK supports half-precision (using IEEE FP16 format) inference for all
floating-point operators. XNNPACK automatically enables half-precision
inference when the following conditions are met:</p>
<ul>
<li>
<p>XNNPACK runs on hardware that natively supports computations in IEEE FP16
format. Currently, this hardware is limited to ARM &amp; ARM64 devices with
ARMv8.2 FP16 arithmetics extension, and includes Android phones starting with
Pixel 3, Galaxy S9 (Snapdragon SoC), Galaxy S10 (Exynos SoC), iOS devices with
A11 or newer SoCs, all Apple Silicon Macs, and Windows ARM64 laptops based with
Snapdragon 850 SoC or newer.</p>
</li>
<li>
<p>The model's "reduced_precision_support" metadata indicates that the model
is compatible with FP16 inference. The metadata can be added during model
conversion using the <code>_experimental_supported_accumulation_type</code> attribute
of the <a href="https://www.tensorflow.org/api_docs/python/tf/lite/TargetSpec">tf.lite.TargetSpec</a>
object:</p>
</li>
</ul>
<p><code>python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
...
converter.target_spec.supported_types = [tf.float16]
converter.target_spec._experimental_supported_accumulation_type = tf.dtypes.float16</code></p>
<p>When the above conditions are met, XNNPACK replace FP32 operators with their
FP16 equivalents, and insert additional operators to convert model inputs
from FP32 to FP16 and convert model outputs back from FP16 to FP32. If the
above conditions are not met, XNNPACK will perform model inference with FP32
calculations.</p>
<p>Additionally, XNNPACK delegate provides an option to force FP16 inference
regardless of model metadata. This option is intended for development workflows,
and in particular for testing end-to-end accuracy of model when FP16 inference
is used. Forcing FP16 inference has several effects:</p>
<ul>
<li>
<p>Besides ARM64 devices with ARMv8.2 FP16 arithmetics extension, forced FP16
inference is supported on x86/x86-64 devices with AVX2 extension in emulation
mode: all elementary floating-point operations are computed in FP32, then
converted to FP16 and back to FP32. Note that such simulation is not bit-exact
equivalent to native FP16 inference, but simulates the effects of restricted
mantissa precision and exponent range in the native FP16 arithmetics.</p>
</li>
<li>
<p>On devices that support neither the native FP16 arithmetics (ARM64 devices
with ARMv8.2 FP16 arithmetics extension), nor emulation (x86/x86-64 devices with
AVX2 extension), inference will fail rather than fall back to FP32.</p>
</li>
<li>
<p>If any floating-point operator offloaded to XNNPACK is not supported for FP16
inference, inference will fail rather than fall back to FP32.</p>
</li>
</ul>
<p>To force FP16 inference, either build the delegate with
<code>--define xnnpack_force_float_precision=fp16</code> option, or add
<code>TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16</code> flag to the
<code>TfLiteXNNPackDelegateOptions.flags</code> bitmask passed into
the <code>TfLiteXNNPackDelegateCreate</code> call:</p>
<p><code>c
TfLiteXNNPackDelegateOptions xnnpack_options =
    TfLiteXNNPackDelegateOptionsDefault();
...
xnnpack_options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
TfLiteDelegate* xnnpack_delegate =
    TfLiteXNNPackDelegateCreate(&amp;xnnpack_options);</code></p>
<p>XNNPACK has full feature parity between FP32 and FP16 operators: all operators
that are supported for FP32 inference are also supported for FP16 inference,
and vice versa. In particular, sparse inference operators are supported for FP16
inference on ARM processors.</p>
<h3>Quantized Operators</h3>
<p>By default, quantized inference in XNNPACK delegate is disabled, and XNNPACK is
used only for floating-point models. Support for quantized inference in XNNPACK
must be enabled by adding extra Bazel flags when building TensorFlow Lite.</p>
<ul>
<li>
<p><code>--define tflite_with_xnnpack_qs8=true</code> flag enables XNNPACK inference for
  quantized operators using signed quantization schema. This schema is used by
  models produced by <a href="https://www.tensorflow.org/model_optimization">Model Optimization
  Toolkit</a> through either
  post-training integer quantization or quantization-aware training.
  Post-training dynamic range quantization is not supported in XNNPACK.</p>
</li>
<li>
<p><code>--define tflite_with_xnnpack_qu8=true</code> flag enables XNNPACK inference for
  quantized operators using unsigned quantization schema, produced via the
  legacy TensorFlow 1.X quantization tooling. This option is experimental and
  may perform suboptimally on mobile processors with NEON DOT product
  instructions.</p>
</li>
</ul>
<p>Below is the list of currently supported quantized operators:</p>
<h4><code>ADD</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format.</li>
<li>Only addition with two inputs is supported.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>CONCATENATION</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format.</li>
<li>Only concatenation with two, three, or four inputs is supported.</li>
</ul>
<h4><code>CONV_2D</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format (bias must be in 32-bit
  quantized format).</li>
<li>Bias is mandatory.</li>
<li>Both filter and bias must be static (use <code>kTfLiteMmapRo</code> allocation type),
  and can use either per-tensor or per-channel quantization parameters.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>DEPTH_TO_SPACE</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format.</li>
<li>Block size must be greater than 1.</li>
</ul>
<h4><code>DEPTHWISE_CONV_2D</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format (bias must be in
  32-bit quantized format).</li>
<li>Bias is mandatory.</li>
<li>Both filter and bias must be static (use <code>kTfLiteMmapRo</code> allocation type),
  and can use either per-tensor or per-channel quantization parameters.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>DEQUANTIZE</code></h4>
<ul>
<li>Input tensor must be in 8-bit quantized format without per-channel
  quantization.</li>
<li>Output tensor must be in 32-bit floating-point format.</li>
</ul>
<h4><code>ELU</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit signed quantized format.</li>
</ul>
<h4><code>FULLY_CONNECTED</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format (bias, if present, must
  be in 32-bit quantized format).</li>
<li>Both filter and bias must be static (use <code>kTfLiteMmapRo</code> allocation type).</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>LEAKY_RELU</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format.</li>
<li>The ratio of input scale to output scale must be within [1/256, 128].</li>
<li>The product of negative slope by the ratio of input scale to output scale
  must be within either [-127.99609375, -1/256] range or [1/256, 128] range.</li>
</ul>
<h4><code>LOGISTIC</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format.</li>
</ul>
<h4><code>MAX_POOL_2D</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format.</li>
<li>1x1 pooling with non-unit stride is not supported.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>MEAN</code></h4>
<ul>
<li>The first input and the output must be 4D tensors in 8-bit quantized format.</li>
<li>The second input (the input with the axes specification) must be static
  (use <code>kTfLiteMmapRo</code> allocation type).</li>
<li>Only [1, 2], [2, 1], and [2] axes specification (i.e. reduction across either
  both spatial dimensions or across the width dimension) is supported.</li>
</ul>
<h4><code>MUL</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>PAD</code></h4>
<ul>
<li>The first input and the output must be in 8-bit quantized format.</li>
<li>The second input (the input with the padding specification) must be static
  (use <code>kTfLiteMmapRo</code> allocation type).</li>
<li>The numbers of padding elements must be non-negative.</li>
</ul>
<h4><code>QUANTIZE</code></h4>
<ul>
<li>Input tensor must be in 32-bit floating-point format or in 8-bit quantized
  format.</li>
<li>Output tensor must be in 8-bit quantized format without per-channel
  quantization.</li>
<li>If inputs are in 8-bit quantized format, they must have the same signedness
  as the outputs, and the ratio of input scale to output scale must be in the
  [2<strong>-8, 2</strong>7] range.</li>
</ul>
<h4><code>RESHAPE</code></h4>
<ul>
<li>The first input and the output must be in 8-bit quantized format.</li>
<li>The second input (the input with the new shape specification) must be either
    static (use <code>kTfLiteMmapRo</code> allocation type), or absent (with the new shape
    specified via <code>ReshapeOptions</code> table).</li>
</ul>
<h4><code>RESIZE_BILINEAR</code></h4>
<ul>
<li>The first input and the output must be 4D tensors in 8-bit quantized format.</li>
<li>The second input (the input with the new shape specification) must be
  static (use <code>kTfLiteMmapRo</code> allocation type).</li>
</ul>
<h4><code>SLICE</code></h4>
<ul>
<li>The first input and the output must be in 8-bit quantized format.</li>
<li>The second and third inputs (the inputs with the slices' begin and size
  specification) must be static (use <code>kTfLiteMmapRo</code> allocation type).</li>
</ul>
<h4><code>SPACE_TO_DEPTH</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format.</li>
<li>Block size must be greater than 1.</li>
</ul>
<h4><code>SPLIT</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format.</li>
<li>Only split into two, three, or four outputs is supported.</li>
</ul>
<h4><code>SUB</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format.</li>
<li>Fused <code>NONE</code>, <code>RELU</code>, <code>RELU_N1_TO_1</code>, and <code>RELU6</code> activations are supported,
  but fused <code>TANH</code> and <code>SIGN_BIT</code> activations are not.</li>
</ul>
<h4><code>TANH</code></h4>
<ul>
<li>Inputs and outputs must be in 8-bit quantized format.</li>
</ul>
<h4><code>TRANSPOSE</code></h4>
<ul>
<li>The first input and the output must be in 8-bit quantized format.</li>
<li>The second input (the input with the permutation specification) must be
  static (use <code>kTfLiteMmapRo</code> allocation type).</li>
</ul>
<h4><code>TRANSPOSE_CONV</code></h4>
<ul>
<li>Input, filter, and output tensors must be in 8-bit quantized format (bias, if
  present, must be in 32-bit quantized format).</li>
<li>Output size, filter and bias (if present) must be static (use
  <code>kTfLiteMmapRo</code> allocation type).</li>
</ul>
<h3>Sparse Inference</h3>
<p>XNNPACK backend supports sparse inference for CNN models described in the
<a href="https://arxiv.org/abs/1911.09723">Fast Sparse ConvNets</a> paper. Sparse
inference is restricted to subgraphs with the following floating-point
operators:</p>
<ul>
<li>Sparse subgraph must store its weights in sparse representation (using
  <code>DENSIFY</code> operators in the TensorFlow Lite schema).</li>
<li>Sparse subgraph must start with a 3x3 stride-2 <code>CONV_2D</code> operator with
  padding 1 on each side, no dilation, and 3 input channels.</li>
<li>Sparse subgraph must end with either a <code>MEAN</code> operator with reduction across
  spatial axes, or a <code>DEPTH_TO_SPACE</code> operator.</li>
<li>Sparse subgraph may contain the following operators:</li>
<li><code>CONV_2D</code> with 1x1 kernel and no padding. At least 2/3rd of filter weights
    in the 1x1 <code>CONV_2D</code> operators across the sparse subgraph must be zeroes
    to enable sparse inference.</li>
<li><code>DEPTHWISE_CONV_2D</code> with 3x3 kernel, stride 1, no dilation, and padding 1
    on each side.</li>
<li><code>DEPTHWISE_CONV_2D</code> with 3x3 kernel, stride 2, no dilation, and padding 1
    on each side.</li>
<li><code>DEPTHWISE_CONV_2D</code> with 5x5 kernel, stride 1, no dilation, and padding 2
    on each side.</li>
<li><code>DEPTHWISE_CONV_2D</code> with 5x5 kernel, stride 2, no dilation, and padding 2
    on each side.</li>
<li><code>RESIZE_BILINEAR</code> operator with output dimensions greater than 1.</li>
<li><code>MEAN</code> operator with reduction across spatial axes.</li>
<li><code>ADD</code> and <code>MUL</code> operators where both inputs are 4D tensors. If one of the
    inputs to <code>ADD</code> or <code>MUL</code> is a constant tensor, it must be representable as
    either a scalar, or a 1D vector.</li>
<li>Unary elementwise operators <code>ABS</code>, <code>CEIL</code>, <code>ELU</code>, <code>FLOOR</code>, <code>HARD_SWISH</code>,
    <code>LEAKY_RELU</code>, <code>LOGISTIC</code>, <code>NEG</code>, <code>RELU</code>, <code>RELU6</code>, <code>RELU_N1_TO_1</code>, <code>ROUND</code>,
    <code>SIGMOID</code>, and <code>SQUARE</code>.</li>
</ul>
<p>Pre-trained <a href="https://github.com/google-research/google-research/tree/master/fastconvnets">Fast Sparse ConvNets models</a>
provide examples that satisfy these constrains.</p>
<h3>Other limitations</h3>
<ul>
<li>Dynamically allocated (with <code>kTfLiteDynamic</code> allocation type) inputs and
  outputs are not supported.</li>
<li>Resizing model inputs (via <code>Interpreter::ResizeInputTensor</code>) is supported, but
  cause a complete reinitialization of the delegate instance, which has
  considerable overhead.</li>
</ul>