<h1>TensorFlow Lite for Swift</h1>
<p><a href="https://www.tensorflow.org/lite/">TensorFlow Lite</a> is TensorFlow's lightweight
solution for Swift developers. It enables low-latency inference of on-device
machine learning models with a small binary size and fast performance supporting
hardware acceleration.</p>
<h2>Build TensorFlow with iOS support</h2>
<p>To build the Swift TensorFlow Lite library on Apple platforms,
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
pod 'TensorFlowLiteSwift'</code></p>
<p>Then, run <code>pod install</code>.</p>
<p>In your Swift files, import the module:</p>
<p><code>swift
import TensorFlowLite</code></p>
<h3>Bazel developers</h3>
<p>In your <code>BUILD</code> file, add the <code>TensorFlowLite</code> dependency to your target:</p>
<p><code>python
swift_library(
  deps = [
      "//tensorflow/lite/swift:TensorFlowLite",
  ],
)</code></p>
<p>In your Swift files, import the module:</p>
<p><code>swift
import TensorFlowLite</code></p>
<p>Build the <code>TensorFlowLite</code> Swift library target:</p>
<p><code>shell
bazel build tensorflow/lite/swift:TensorFlowLite</code></p>
<p>Build the <code>Tests</code> target:</p>
<p><code>shell
bazel test tensorflow/lite/swift:Tests --swiftcopt=-enable-testing</code></p>
<p>Note: <code>--swiftcopt=-enable-testing</code> is required for optimized builds (<code>-c opt</code>).</p>
<h4>Generate the Xcode project using Tulsi</h4>
<p>Open the <code>//tensorflow/lite/swift/TensorFlowLite.tulsiproj</code> using
the <a href="https://github.com/bazelbuild/tulsi">TulsiApp</a>
or by running the
<a href="https://github.com/bazelbuild/tulsi/blob/master/src/tools/generate_xcodeproj.sh"><code>generate_xcodeproj.sh</code></a>
script from the root <code>tensorflow</code> directory:</p>
<p><code>shell
generate_xcodeproj.sh --genconfig tensorflow/lite/swift/TensorFlowLite.tulsiproj:TensorFlowLite --outputfolder ~/path/to/generated/TensorFlowLite.xcodeproj</code></p>