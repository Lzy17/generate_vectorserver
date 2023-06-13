<h1>Tensorflow Lite Core ML Delegate</h1>
<p>TensorFlow Lite Core ML Delegate enables running TensorFlow Lite models on
<a href="https://developer.apple.com/documentation/coreml">Core ML framework</a>,
which results in faster model inference on iOS devices.</p>
<p>[TOC]</p>
<h2>Supported iOS versions and processors</h2>
<ul>
<li>iOS 12 and later. In the older iOS versions, Core ML delegate will
  automatically fallback to CPU.</li>
<li>When running on iPhone Xs and later, it will use Neural Engine for faster
  inference.</li>
</ul>
<h2>Update code to use Core ML delegate</h2>
<h3>Swift</h3>
<p>Initialize TensorFlow Lite interpreter with Core ML delegate.</p>
<p><code>swift
let coreMlDelegate = CoreMLDelegate()
let interpreter = try Interpreter(modelPath: modelPath,
                                  delegates: [coreMLDelegate])</code></p>
<h3>Objective-C++</h3>
<h4>Interpreter initialization</h4>
<p>Include <code>coreml_delegate.h</code>.</p>
<p>```objectivec++</p>
<h1>include "tensorflow/lite/experimental/delegates/coreml/coreml_delegate.h"</h1>
<p>```</p>
<p>Modify code following interpreter initialization to apply delegate.</p>
<p>```objectivec++
// initializer interpreter with model.
tflite::InterpreterBuilder(*model, resolver)(&amp;interpreter);</p>
<p>// Add following section to use Core ML delegate.
TfLiteCoreMlDelegateOptions options = {};
delegate = TfLiteCoreMlDelegateCreate(&amp;options);
interpreter-&gt;ModifyGraphWithDelegate(delegate);</p>
<p>// ...
```</p>
<h4>Disposal</h4>
<p>Add this code to the section where you dispose of the delegate (e.g. <code>dealloc</code>
of class).</p>
<p><code>objectivec++
TfLiteCoreMlDelegateDelete(delegate);</code></p>
<h2>Supported ops</h2>
<p>Following ops are supported by the Core ML delegate.</p>
<ul>
<li>Add<ul>
<li>Only certain shapes are broadcastable. In Core ML tensor layout,
    following tensor shapes are broadcastable. <code>[B, C, H, W]</code>, <code>[B, C, 1,
    1]</code>, <code>[B, 1, H, W]</code>, <code>[B, 1, 1, 1]</code>.</li>
</ul>
</li>
<li>AveragePool2D</li>
<li>Concat</li>
<li>Conv2D<ul>
<li>Weights and bias should be constant.</li>
</ul>
</li>
<li>DepthwiseConv2D<ul>
<li>Weights and bias should be constant.</li>
</ul>
</li>
<li>FullyConnected (aka Dense or InnerProduct)<ul>
<li>Weights and bias (if present) should be constant.</li>
<li>Only supports single-batch case. Input dimensions should be 1, except
    the last dimension.</li>
</ul>
</li>
<li>Hardswish</li>
<li>Logistic (aka Sigmoid)</li>
<li>MaxPool2D</li>
<li>MirrorPad<ul>
<li>Only 4D input with <code>REFLECT</code> mode is supported. Padding should be
    constant, and is only allowed for H and W dimensions.</li>
</ul>
</li>
<li>Mul<ul>
<li>Only certain shapes are broadcastable. In Core ML tensor layout,
    following tensor shapes are broadcastable. <code>[B, C, H, W]</code>, <code>[B, C, 1,
    1]</code>, <code>[B, 1, H, W]</code>, <code>[B, 1, 1, 1]</code>.</li>
</ul>
</li>
<li>Pad and PadV2<ul>
<li>Only 4D input is supported. Padding should be constant, and is only
    allowed for H and W dimensions.</li>
</ul>
</li>
<li>Relu</li>
<li>ReluN1To1</li>
<li>Relu6</li>
<li>Reshape<ul>
<li>Only supported when target Core ML version is 2, not supported when
    targeting Core ML 3.</li>
</ul>
</li>
<li>ResizeBilinear</li>
<li>SoftMax</li>
<li>Tanh</li>
<li>TransposeConv<ul>
<li>Weights should be constant.</li>
</ul>
</li>
</ul>
<h2>FAQ</h2>
<ul>
<li>Does Core ML delegate support fallback to CPU if a graph contains unsupported
  ops?</li>
<li>Yes.</li>
<li>Does Core ML delegate work on iOS Simulator?</li>
<li>Yes. The library includes x86 and x86_64 targets so it can run on
    a simulator, but you will not see performance boost over CPU.</li>
<li>Does TensorFlow Lite and Core ML delegate support macOS?</li>
<li>TensorFlow Lite is only tested on iOS but not macOS.</li>
<li>Are custom TF Lite ops supported?</li>
<li>No, CoreML delegate does not support custom ops and they will fallback to
    CPU.</li>
</ul>
<h2>Appendix</h2>
<h3>Core ML delegate Swift API</h3>
<p><code>``swift
/// A delegate that uses the</code>Core ML<code>framework for performing
/// TensorFlow Lite graph operations.
///
/// - Important: This is an experimental interface that is subject to change.
public final class CoreMLDelegate: Delegate {
 /// The configuration options for the</code>CoreMLDelegate`.
 public let options: Options</p>
<p>// Conformance to the <code>Delegate</code> protocol.
 public private(set) var cDelegate: CDelegate</p>
<ul>
<li>/// Creates a new instance configured with the given <code>options</code>.
 ///
 /// - Parameters:
 ///   - options: Configurations for the delegate. The default is a new instance of
 ///       <code>CoreMLDelegate.Options</code> with the default configuration values.
 public init(options: Options = Options()) {
   self.options = options
   var delegateOptions = TfLiteCoreMlDelegateOptions()
   cDelegate = TfLiteCoreMlDelegateCreate(&amp;delegateOptions)
 }</li>
</ul>
<p>deinit {
   TfLiteCoreMlDelegateDelete(cDelegate)
 }
}</p>
<p>extension CoreMLDelegate {
 /// Options for configuring the <code>CoreMLDelegate</code>.
 public struct Options: Equatable, Hashable {
   /// Creates a new instance with the default values.
   public init() {}
 }
}
```</p>
<h3>Core ML delegate C++ API</h3>
<p>```c++
typedef struct {
  // Only create delegate when Neural Engine is available on the device.
  TfLiteCoreMlDelegateEnabledDevices enabled_devices;
  // Specifies target Core ML version for model conversion.
  // Core ML 3 come with a lot more ops, but some ops (e.g. reshape) is not
  // delegated due to input rank constraint.
  // if not set to one of the valid versions, the delegate will use highest
  // version possible in the platform.
  // Valid versions: (2, 3)
  int coreml_version;
  // This sets the maximum number of Core ML delegates created.
  // Each graph corresponds to one delegated node subset in the
  // TFLite model. Set this to 0 to delegate all possible partitions.
  int max_delegated_partitions;
  // This sets the minimum number of nodes per partition delegated with
  // Core ML delegate. Defaults to 2.
  int min_nodes_per_partition;</p>
<h1>ifdef TFLITE_DEBUG_DELEGATE</h1>
<p>// This sets the index of the first node that could be delegated.
  int first_delegate_node_index;
  // This sets the index of the last node that could be delegated.
  int last_delegate_node_index;</p>
<h1>endif</h1>
<p>} TfLiteCoreMlDelegateOptions;</p>
<p>// Return a delegate that uses CoreML for ops execution.
// Must outlive the interpreter.
TfLiteDelegate<em> TfLiteCoreMlDelegateCreate(
   const TfLiteCoreMlDelegateOptions</em> options);</p>
<p>// Do any needed cleanup and delete 'delegate'.
void TfLiteCoreMlDelegateDelete(TfLiteDelegate* delegate);
```</p>