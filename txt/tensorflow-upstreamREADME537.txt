<h1>TensorFlow Lite for Objective-C</h1>
<p><a href="https://www.tensorflow.org/lite/">TensorFlow Lite</a> is TensorFlow's lightweight
solution for Objective-C developers. It enables low-latency inference of
on-device machine learning models with a small binary size and fast performance
supporting hardware acceleration.</p>
<h2>Build TensorFlow with iOS support</h2>
<p>To build the Objective-C TensorFlow Lite library on Apple platforms,
<a href="https://www.tensorflow.org/install/source#setup_for_linux_and_macos">install from source</a>
or <a href="https://github.com/tensorflow/tensorflow">clone the GitHub repo</a>.
Then, configure TensorFlow by navigating to the root directory and executing the
<code>configure.py</code> script:</p>
<p><code>shell
python configure.py</code></p>
<p>Follow the prompts and when asked to build TensorFlow with iOS support, enter <code>y</code>.</p>
<h3>CocoaPods developers</h3>
<p>Add the TensorFlow Lite pod to your <code>Podfile</code>:</p>
<p><code>ruby
pod 'TensorFlowLiteObjC'</code></p>
<p>Then, run <code>pod install</code>.</p>
<p>In your Objective-C files, import the umbrella header:</p>
<p>```objectivec</p>
<h1>import "TFLTensorFlowLite.h"</h1>
<p>```</p>
<p>Or, the module if you set <code>CLANG_ENABLE_MODULES = YES</code> in your Xcode project:</p>
<p><code>objectivec
@import TFLTensorFlowLite;</code></p>
<p>Note: To import the TensorFlow Lite module in your Objective-C files, you must
also include <code>use_frameworks!</code> in your <code>Podfile</code>.</p>
<h3>Bazel developers</h3>
<p>In your <code>BUILD</code> file, add the <code>TensorFlowLite</code> dependency to your target:</p>
<p><code>python
objc_library(
    deps=[
        "//tensorflow/lite/objc:TensorFlowLite",
    ],)</code></p>
<p>In your Objective-C files, import the umbrella header:</p>
<p>```objectivec</p>
<h1>import "TFLTensorFlowLite.h"</h1>
<p>```</p>
<p>Or, the module if you set <code>CLANG_ENABLE_MODULES = YES</code> in your Xcode project:</p>
<p><code>objectivec
@import TFLTensorFlowLite;</code></p>
<p>Build the <code>TensorFlowLite</code> Objective-C library target:</p>
<p><code>shell
bazel build tensorflow/lite/objc:TensorFlowLite</code></p>
<p>Build the <code>tests</code> target:</p>
<p><code>shell
bazel test tensorflow/lite/objc:tests</code></p>
<h4>Generate the Xcode project using Tulsi</h4>
<p>Open the <code>//tensorflow/lite/objc/TensorFlowLite.tulsiproj</code> using
the <a href="https://github.com/bazelbuild/tulsi">TulsiApp</a>
or by running the
<a href="https://github.com/bazelbuild/tulsi/blob/master/src/tools/generate_xcodeproj.sh"><code>generate_xcodeproj.sh</code></a>
script from the root <code>tensorflow</code> directory:</p>
<p><code>shell
generate_xcodeproj.sh --genconfig tensorflow/lite/objc/TensorFlowLite.tulsiproj:TensorFlowLite --outputfolder ~/path/to/generated/TensorFlowLite.xcodeproj</code></p>