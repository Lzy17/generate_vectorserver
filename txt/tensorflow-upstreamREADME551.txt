<h1>TFLite Model Benchmark Tool with Android Apk</h1>
<h2>Description</h2>
<p>This Android benchmark app is a simple wrapper around the TensorFlow Lite
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark">command-line benchmark utility</a>.</p>
<p>Pushing and executing binaries directly on an Android device is a valid approach
to benchmarking, but it can result in subtle (but observable) differences in
performance relative to execution within an actual Android app. In particular,
Android's scheduler tailors behavior based on thread and process priorities,
which differ between a foreground Activity/Application and a regular background
binary executed via <code>adb shell ...</code>. This tailored behavior is most evident when
enabling multi-threaded CPU execution with TensorFlow Lite.</p>
<p>To that end, this app offers perhaps a more faithful view of runtime performance
that developers can expect when deploying TensorFlow Lite with their
application.</p>
<h2>To build/install/run</h2>
<p>(0) Refer to
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/android/test
to edit the <code>WORKSPACE</code> to configure the android NDK/SDK.</p>
<p>(1) Build for your specific platform, e.g.:</p>
<p><code>bazel build -c opt \
  --config=android_arm64 \
  tensorflow/lite/tools/benchmark/android:benchmark_model</code></p>
<p>(Optional) To enable Hexagon delegate with <code>--use_hexagon=true</code> option, you can
download and install the libraries as the guided in [hexagon delegate]
(https://www.tensorflow.org/lite/performance/hexagon_delegate#step_2_add_hexagon_libraries_to_your_android_app)
page. For example, if you installed the libraries at third_party/hexagon_nn_skel
and created third_party/hexagon_nn_skel/BUILD with a build target,</p>
<p><code>filegroup(
    name = "libhexagon_nn_skel",
    srcs = glob(["*.so"]),
)</code></p>
<p>you need to modify tflite_hexagon_nn_skel_libraries macro in
tensorflow/lite/special_rules.bzl to specify the build target.</p>
<p><code>return ["//third_party/hexagon_nn_skel:libhexagon_nn_skel"]</code></p>
<p>(2) Connect your phone. Install the benchmark APK to your phone with adb:</p>
<p><code>adb install -r -d -g bazel-bin/tensorflow/lite/tools/benchmark/android/benchmark_model.apk</code></p>
<p>Note: Make sure to install with "-g" option to grant the permission for reading
external storage.</p>
<p>(3) Push the compute graph that you need to test.</p>
<p><code>adb push mobilenet_quant_v1_224.tflite /data/local/tmp</code></p>
<p>(4) Run the benchmark. Additional command-line flags are documented
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/README.md">here</a>
and can be appended to the <code>args</code> string alongside the required <code>--graph</code> flag
(note that all args must be nested in the single quoted string that follows the
args key).</p>
<p><code>adb shell am start -S \
  -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \
  --es args '"--graph=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --num_threads=4"'</code></p>
<p>(5) The results will be available in Android's logcat, e.g.:</p>
<p>```
adb logcat | grep "Inference timings in us"</p>
<p>... tflite  : Inference timings in us: Init: 1007529, First inference: 4098, Warmup (avg): 1686.59, Inference (avg): 1687.92
```</p>
<h2>To trace Tensorflow Lite internals including operator invocation</h2>
<p>The steps described here follows the method of
https://developer.android.com/topic/performance/tracing/on-device. Refer to the
page for more detailed information.</p>
<p>(0)-(3) Follow the steps (0)-(3) of <a href="#to-buildinstallrun">build/install/run</a>
section.</p>
<p>(4) Enable platform tracing.</p>
<p><code>adb shell setprop debug.tflite.trace 1</code></p>
<p>(5) Set up Quick Settings tile for System Tracing app on your device. Follow the
<a href="https://developer.android.com/topic/performance/tracing/on-device#set-up-tile">instruction</a>.
The System Tracing tile will be added to the Quick Settings panel.</p>
<p>Optionally, you can set up other configurations for tracing from the app menu.
Refer to the
<a href="https://developer.android.com/topic/performance/tracing/on-device#app-menu">guide</a>
for more information.</p>
<p>(6) Tap the System Tracing tile, which has the label "Record trace". The tile
becomes enabled, and a persistent notification appears to notify you that the
system is now recording a trace.</p>
<p>(7) Run the benchmark with platform tracing enabled.</p>
<p><code>adb shell am start -S \
  -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \
  --es args '"--graph=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --num_threads=4"'</code></p>
<p>(8) Wait until the benchmark finishes. It can be checked from Android log
messages, e.g.,</p>
<p>```
adb logcat | grep "Inference timings in us"</p>
<p>... tflite  : Inference timings in us: Init: 1007529, First inference: 4098, Warmup (avg): 1686.59, Inference (avg): 1687.92
```</p>
<p>(9) Stop tracing by tapping either the System Tracing tile in the Quick Settings
panel or on the System Tracing notification. The system displays a new
notification that contains the message "Saving trace". When saving is complete,
the system dismisses the notification and displays a third notification "Trace
saved", confirming that your trace has been saved and that you're ready to share
the system trace.</p>
<p>(10)
<a href="https://developer.android.com/topic/performance/tracing/on-device#share-trace">Share</a>
a trace file,
<a href="https://developer.android.com/topic/performance/tracing/on-device#converting_between_trace_formats">convert</a>
between tracing formats and
<a href="https://developer.android.com/topic/performance/tracing/on-device#create-html-report">create</a>
an HTML report. Note that, the captured tracing file format is either in
Perfetto format or in Systrace format depending on the Android version of your
device. Select the appropriate method to handle the generated file.</p>
<p>(11) Disable platform tracing.</p>
<p><code>adb shell setprop debug.tflite.trace 0</code></p>