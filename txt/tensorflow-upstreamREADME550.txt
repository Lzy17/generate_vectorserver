<h1>TensorFlow Lite Delegate Performance Benchmark (Android APK)</h1>
<h2>Description</h2>
<p>This Android Delegate Performance Benchmark (DPB) app is a simple wrapper around
the TensorFlow Lite
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark">benchmark tool</a>
and
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/acceleration/mini_benchmark">MiniBenchmark</a>
with the focus on testing Tensorflow Lite delegates that implement stable
delegate ABI.</p>
<p>Development of TensorFlow Lite delegates needs both accuracy and latency
evaluations for catching potential performance regressions. Pushing and
executing both latency and accuracy testing binaries directly on an Android
device is a valid approach to benchmarking, but it can result in subtle (but
observable) differences in performance relative to execution within an actual
Android app. In particular, Android's scheduler tailors behavior based on thread
and process priorities, which differ between a foreground Activity/Application
and a regular background binary executed via <code>adb shell ...</code>.</p>
<p>In addition to that, having multiple benchmarking apps for different performance
metric evaluations could potentially cost development effort unnecessarily.</p>
<p>To those ends, this app offers a more faithful view of runtime performance
(accuracy and latency) that developers can expect when using TensorFlow Lite
delegates with Android apps, and the app provides a single entrypoint to various
performance metrics to avoid the need to switch between different benchmarking
apps.</p>
<h2>To build/install/run</h2>
<h3>Build</h3>
<ol>
<li>
<p>Clone the TensorFlow repo with</p>
<p><code>git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git</code></p>
<p>Note: --recurse-submodules is necessary to prevent some issues with protobuf
compilation.</p>
</li>
<li>
<p>Refer to
    <a href="https://www.tensorflow.org/lite/android/lite_build#set_up_build_environment_without_docker">this page</a>
    for setting up a development environment. Although there are several
    practical tips:</p>
<ul>
<li>When installing Bazel, for Ubuntu Linux, <code>sudo apt update &amp;&amp; sudo apt
    install bazel</code> may be the easiest way. However sometimes you may need
    <code>sudo apt update &amp;&amp; sudo apt install bazel-5.3.0</code> if prompted.</li>
<li>When installing Android NDK and SDK, using Android Studio's SDK Manager
    may be the easiest way.</li>
<li>Run the <code>./configure</code> script in the root TensorFlow checkout directory,
    and answer "Yes" when the script asks to interactively configure the
    <code>./WORKSPACE</code> for Android builds.</li>
<li>The versions which we have verified are working:<ul>
<li>Android NDK version: 21.4.7075529<ul>
<li>Provide the value as part of a path when the <code>./configure</code>
    script asks to specify the home path of the Android NDK.</li>
</ul>
</li>
<li>Android NDK API level: 26<ul>
<li>Provide the value when the <code>./configure</code> script asks to specify
    the Android NDK API level.</li>
</ul>
</li>
<li>Android SDK API level: 33<ul>
<li>Provide the value when the <code>./configure</code> script asks to specify
    the Android SDK API level.</li>
</ul>
</li>
<li>Android build tools version: 30.0.0<ul>
<li>Provide the value when the <code>./configure</code> script asks to specify
    the Android build tools version.</li>
</ul>
</li>
</ul>
</li>
</ul>
</li>
<li>
<p>Build for your specific platform, e.g.:</p>
<p><code>bazel build -c opt \
  --config=android_arm64 \
  tensorflow/lite/tools/benchmark/experimental/delegate_performance/android:delegate_performance_benchmark</code></p>
</li>
</ol>
<h3>Install</h3>
<ol>
<li>
<p>Connect to a physical device. Install the benchmark APK with adb:</p>
<p><code>adb install -r -d -g bazel-bin/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/delegate_performance_benchmark.apk</code></p>
<p>Note: Make sure to install with "-g" option to grant the permission for
reading external storage.</p>
</li>
</ol>
<h3>Run</h3>
<h4>Benchmarking with stable delegates</h4>
<p>The delegate-under-test must implement the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h">stable_delegate_interface</a>
API. The stable delegate provider dynamically loads stable delegate symbols from
the provided binary (shared object) file. In order to use Delegate Performance
Benchmark with a stable delegate, you would need to push the shared object file
to the file directory of Delegate Performance Benchmark:
<code>/data/data/org.tensorflow.lite.benchmark.delegateperformance/files/</code>.</p>
<ol>
<li>
<p>Build and push the stable delegate binary that you want to test. Here we use
    the
    <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate">sample stable delegate</a>
    as an example.</p>
<p>```
bazel build -c opt \
  --config=android_arm64 \
  tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate</p>
<h1>Set the permissions so that we can overwrite a previously installed delegate.</h1>
<p>chmod 755 bazel-bin/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so</p>
<h1>Ensure the delegateperformance files path exists.</h1>
<p>adb shell run-as org.tensorflow.lite.benchmark.delegateperformance mkdir -p /data/data/org.tensorflow.lite.benchmark.delegateperformance/files</p>
<h1>Install the sample delegate.</h1>
<p>adb push \
  bazel-bin/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so \
  /data/local/tmp/
adb shell run-as org.tensorflow.lite.benchmark.delegateperformance \
  cp /data/local/tmp/libtensorflowlite_sample_stable_delegate.so \
    /data/data/org.tensorflow.lite.benchmark.delegateperformance/files/
```</p>
</li>
<li>
<p>Dump the test sample delegate settings file on device. Example command:</p>
<p><code>adb shell 'echo "{
  \"delegate\": \"NONE\",  // Replace NONE with the test target delegate type.
  \"stable_delegate_loader_settings\": {
    \"delegate_path\": \"/data/data/org.tensorflow.lite.benchmark.delegateperformance/files/libtensorflowlite_sample_stable_delegate.so\"
  }
  // Add concrete delegate settings for the test target delegate.
}
"&gt; /data/local/tmp/stable_delegate_settings.json'</code></p>
</li>
</ol>
<h4>Supported models</h4>
<p>Currently DPB uses a <code>mobilenet_v1_1.0_224.tflite</code> and
<code>mobilenet_quant_v1_224.tflite</code> model for latency and accuracy benchmarking. The
TF Lite model files are bundled into the app during the build process. We plan
to expand the supported models based on future use cases.</p>
<p>Note: The sample stable delegate provided here only supports ADD and SUB
operations thus aforementioned mobilenet models would not actually be delegated.
To test your own delegate against the models, please update
<code>stable_delegate_loader_settings</code> with your delegate path. To get feedback early
in the development process, e.g. while working towards supporting more OPs, you
can run the <code>benchmark_model</code> tool, which supports stable delegates and can be
supplied with arbitrary models via the <code>--graph</code> CLI parameter. See
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/README.md#tf-lite-benchmark-tool">this document</a>
which shows how to run a model with ADD operations through the sample stable
delegate.</p>
<h4>Latency benchmarking</h4>
<h5>Options</h5>
<ul>
<li><code>tflite_settings_files</code>: <code>str</code> (required) the comma-delimited paths to the
    JSON-encoded delegate <code>TFLiteSettings</code> file(s), which is defined in
    <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/acceleration/configuration/configuration.proto">configuration.proto</a>.</li>
<li>Additional optional command-line flags are documented
    <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/README.md">here</a>
    and can be appended to the <code>args</code> string (note that all args must be nested
    in the single quoted string that follows the args key).</li>
</ul>
<h5>Recommendation Criteria</h5>
<p>The latency benchmark generates a <code>PASS</code>, <code>PASS_WITH_WARNING</code>, or <code>FAIL</code>
recommendation by checking if the regressions of the below metrics for each pair
of the test target delegate and a reference delegate breach the thresholds:</p>
<ol>
<li>Startup overhead latency: the combined overhead start from initialization to
    inferences with stable latency. It is calculated as <code>initialization time +
    average warmup time - average inference time</code>.</li>
<li>Average inference latency: average time for the inferences after warmup in
    the benchmark run.</li>
</ol>
<p>When the test target delegate type is the same as the reference delegate, the
checks are more strict. Otherwise, the checks are relaxed. Please see
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkResultType.java">BenchmarkResultType.java</a>
for the meanings of <code>PASS</code>, <code>PASS_WITH_WARNING</code> and <code>FAIL</code>.</p>
<h5>Steps</h5>
<ol>
<li>
<p>Run the latency benchmark by supplying the settings file via the required
    <code>--tflite_settings_files</code> flag.</p>
<p><code>adb shell "am start -S \
-n org.tensorflow.lite.benchmark.delegateperformance/org.tensorflow.lite.benchmark.delegateperformance.BenchmarkLatencyActivity \
--esa --tflite_settings_files '/data/local/tmp/stable_delegate_settings.json'"</code></p>
</li>
<li>
<p>The results will be available in Android logcat as well as the app's file
    directory, e.g.:</p>
<p>```
adb logcat -c &amp;&amp; adb logcat -v color  | grep "Inference timings in us"</p>
<p>... tflite  : Inference timings in us: Init: 5811, First inference: 67743, Warmup (avg): 65539, Inference (avg): 65155.5
```</p>
<p>The tool also shows overall results.</p>
<p><code>adb logcat -c &amp;&amp; adb logcat -v color  | grep 'Latency benchmark result'</code></p>
<p>which might show output like the following.</p>
<p><code>... TfLiteLatencyImpl: Latency benchmark result for /data/local/tmp/stable_delegate_settings.json: PASS</code></p>
<p>For a summarized view, run</p>
<p><code>adb shell run-as org.tensorflow.lite.benchmark.delegateperformance "cat /data/user/0/org.tensorflow.lite.benchmark.delegateperformance/files/delegate_performance_result/latency/report.html" &gt; /tmp/dpb-latency.html &amp;&amp; xdg-open /tmp/dpb-latency.html</code></p>
<p>It would open a page in the browser like the following:</p>
<p>Summary | FAIL
------- | ----</p>
<p>Model                      | Metric                                         | Delegate: NONE (/data/local/tmp/stable_delegate_settings.json) | Delegate: NONE (default_delegate) | Change    | Status
-------------------------- | ---------------------------------------------- | -------------------------------------------------------------- | --------------------------------- | --------- | ------
mobilenet_v1_1.0_224       | model_size_megabyte                            | -1.0E-6                                                        | -1.0E-6                           | 0.0%      | N/A
mobilenet_v1_1.0_224       | initialization_latency_us                      | 515667.0                                                       | 844938.0                          | -39.0%    | N/A
mobilenet_v1_1.0_224       | warmup_latency_average_us                      | 334263.5                                                       | 1666704.0                         | -79.9%    | N/A
mobilenet_v1_1.0_224       | warmup_latency_min_us                          | 318494.0                                                       | 1666704.0                         | -80.9%    | N/A
mobilenet_v1_1.0_224       | warmup_latency_max_us                          | 350033.0                                                       | 1666704.0                         | -79.0%    | N/A
mobilenet_v1_1.0_224       | warmup_latency_standard_deviation_us           | 15769.0                                                        | 0.0                               | Infinity% | N/A
mobilenet_v1_1.0_224       | inference_latency_average_us                   | 316702.6                                                       | 630715.5                          | -49.8%    | PASS
mobilenet_v1_1.0_224       | inference_latency_min_us                       | 308218.0                                                       | 314117.0                          | -1.9%     | N/A
mobilenet_v1_1.0_224       | inference_latency_max_us                       | 338494.0                                                       | 1601144.0                         | -78.9%    | N/A
mobilenet_v1_1.0_224       | inference_latency_standard_deviation_us        | 4896.0                                                         | 347805.0                          | -98.6%    | N/A
mobilenet_v1_1.0_224       | initialization_memory_max_rss_mebibyte         | 0.0                                                            | 34.48828                          | -100.0%   | N/A
mobilenet_v1_1.0_224       | initialization_memory_total_allocated_mebibyte | 0.0                                                            | 0.0                               | 0.0%      | N/A
mobilenet_v1_1.0_224       | initialization_memory_in_use_mebibyte          | 26.140594                                                      | 21.560455                         | 21.2%     | N/A
mobilenet_v1_1.0_224       | overall_memory_max_rss_mebibyte                | 0.0                                                            | 50.371094                         | -100.0%   | N/A
mobilenet_v1_1.0_224       | overall_memory_total_allocated_mebibyte        | 0.0                                                            | 0.0                               | 0.0%      | N/A
mobilenet_v1_1.0_224       | overall_memory_in_use_mebibyte                 | 28.22168                                                       | 23.295578                         | 21.1%     | N/A
mobilenet_v1_1.0_224       | startup_overhead_latency_us                    | 533227.9                                                       | 1880926.5                         | -71.7%    | PASS
mobilenet_v1_1.0_224       | delegate_summary                               |                                                                |                                   |           | PASS (strict)
mobilenet_v1_1.0_224       | model_summary                                  | PASS                                                           |                                   |           |
mobilenet_v1_1.0_224_quant | model_size_megabyte                            | -1.0E-6                                                        | -1.0E-6                           | 0.0%      | N/A
mobilenet_v1_1.0_224_quant | initialization_latency_us                      | 25318.0                                                        | 8271.0                            | 206.1%    | N/A
mobilenet_v1_1.0_224_quant | warmup_latency_average_us                      | 189565.0                                                       | 188034.0                          | 0.8%      | N/A
mobilenet_v1_1.0_224_quant | warmup_latency_min_us                          | 181333.0                                                       | 175592.0                          | 3.3%      | N/A
mobilenet_v1_1.0_224_quant | warmup_latency_max_us                          | 199285.0                                                       | 199388.0                          | -0.1%     | N/A
mobilenet_v1_1.0_224_quant | warmup_latency_standard_deviation_us           | 7404.0                                                         | 9745.0                            | -24.0%    | N/A
mobilenet_v1_1.0_224_quant | inference_latency_average_us                   | 178905.2                                                       | 178897.69                         | 0.0%      | PASS_WITH_WARNING
mobilenet_v1_1.0_224_quant | inference_latency_min_us                       | 170126.0                                                       | 170102.0                          | 0.0%      | N/A
mobilenet_v1_1.0_224_quant | inference_latency_max_us                       | 200089.0                                                       | 193949.0                          | 3.2%      | N/A
mobilenet_v1_1.0_224_quant | inference_latency_standard_deviation_us        | 6355.0                                                         | 6387.0                            | -0.5%     | N/A
mobilenet_v1_1.0_224_quant | initialization_memory_max_rss_mebibyte         | 0.0                                                            | 0.0                               | 0.0%      | N/A
mobilenet_v1_1.0_224_quant | initialization_memory_total_allocated_mebibyte | 0.0                                                            | 0.0                               | 0.0%      | N/A
mobilenet_v1_1.0_224_quant | initialization_memory_in_use_mebibyte          | 1.4762268                                                      | 1.4715118                         | 0.3%      | N/A
mobilenet_v1_1.0_224_quant | overall_memory_max_rss_mebibyte                | 0.0                                                            | 0.0                               | 0.0%      | N/A
mobilenet_v1_1.0_224_quant | overall_memory_total_allocated_mebibyte        | 0.0                                                            | 0.0                               | 0.0%      | N/A
mobilenet_v1_1.0_224_quant | overall_memory_in_use_mebibyte                 | 3.3774261                                                      | 3.38266                           | -0.2%     | N/A
mobilenet_v1_1.0_224_quant | startup_overhead_latency_us                    | 35977.797                                                      | 17407.312                         | 106.7%    | FAIL
mobilenet_v1_1.0_224_quant | delegate_summary                               |                                                                |                                   |           | FAIL (strict)
mobilenet_v1_1.0_224_quant | model_summary                                  | FAIL                                                           |                                   |           |</p>
</li>
</ol>
<h4>Accuracy benchmarking</h4>
<h5>Options</h5>
<ul>
<li><code>tflite_settings_files</code>: <code>str</code> (required) the comma-delimited paths to the
    JSON-encoded delegate <code>TFLiteSettings</code> file(s), which is defined in
    <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/acceleration/configuration/configuration.proto">configuration.proto</a>.
    The first path is the test target delegate and all other paths are treated
    as reference delegates. The test target delegate will be compared against
    each reference delegate.</li>
</ul>
<h5>Recommendation Criteria</h5>
<p>The accuracy benchmark delegates the accuracy metric threshold checks to the
metric scripts, which are embedded together with the test input inside the
models. The metric scripts generate an "ok" result by aggregating the outcomes
for every model and every delegate. The accuracy benchmark generates a <code>PASS</code>,
<code>PASS_WITH_WARNING</code>, or <code>FAIL</code> recommendation by aggregating the "ok" results.</p>
<p>When the test target delegate type is the same as the reference delegate, the
checks are more strict. Otherwise, the checks are relaxed. Please see
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkResultType.java">BenchmarkResultType.java</a>
for the meanings of <code>PASS</code>, <code>PASS_WITH_WARNING</code> and <code>FAIL</code>.</p>
<h5>Steps</h5>
<ol>
<li>
<p>Run the accuracy benchmark by supplying the settings file via the required
    <code>--tflite_settings_files</code> flag.</p>
<p><code>adb shell "am start -S \
-n org.tensorflow.lite.benchmark.delegateperformance/org.tensorflow.lite.benchmark.delegateperformance.BenchmarkAccuracyActivity \
--esa --tflite_settings_files '/data/local/tmp/stable_delegate_settings.json'"</code></p>
</li>
<li>
<p>The results will be available in Android logcat, e.g.:</p>
<p>```
adb logcat -c &amp;&amp; adb logcat -v color | grep "tflite"</p>
<p>... tflite  : tflite  :   accuracy: ok
```</p>
<p>For a summarized view, run</p>
<p><code>adb shell run-as org.tensorflow.lite.benchmark.delegateperformance "cat /data/user/0/org.tensorflow.lite.benchmark.delegateperformance/files/delegate_performance_result/accuracy/report.html" &gt; /tmp/dpb-accuracy.html &amp;&amp; xdg-open /tmp/dpb-accuracy.html</code></p>
<p>It would open a page in the browser like the following:</p>
<p>Summary | PASS
------- | ----</p>
<p>Model                                      | Metric                           | Delegate: NONE (/data/local/tmp/stable_delegate_settings.json) | Delegate: NONE (default_delegate) | Change | Status
------------------------------------------ | -------------------------------- | -------------------------------------------------------------- | --------------------------------- | ------ | ------
mobilenet_v1_1.0_224_quant_with_validation | mse(average)                     | 1.917638E-6                                                    | 1.917638E-6                       | 0.0%   | N/A
mobilenet_v1_1.0_224_quant_with_validation | symmetric_kl_divergence(average) | 0.049423933                                                    | 0.049423933                       | 0.0%   | N/A
mobilenet_v1_1.0_224_quant_with_validation | ok                               | 0.0                                                            | 0.0                               | N/A    | PASS
mobilenet_v1_1.0_224_quant_with_validation | max_memory_kb                    | 0.0                                                            | 0.0                               | 0.0%   | N/A
mobilenet_v1_1.0_224_quant_with_validation | delegate_summary                 |                                                                |                                   |        | PASS (strict)
mobilenet_v1_1.0_224_quant_with_validation | model_summary                    | PASS                                                           |                                   |        |
mobilenet_v1_1.0_224_with_validation       | mse(average)                     | 1.0577066E-16                                                  | 1.0577066E-16                     | 0.0%   | N/A
mobilenet_v1_1.0_224_with_validation       | symmetric_kl_divergence(average) | 7.2540787E-9                                                   | 7.2540787E-9                      | 0.0%   | N/A
mobilenet_v1_1.0_224_with_validation       | ok                               | 0.0                                                            | 0.0                               | N/A    | PASS
mobilenet_v1_1.0_224_with_validation       | max_memory_kb                    | 0.0                                                            | 0.0                               | 0.0%   | N/A
mobilenet_v1_1.0_224_with_validation       | delegate_summary                 |                                                                |                                   |        | PASS (strict)
mobilenet_v1_1.0_224_with_validation       | model_summary                    | PASS                                                           |                                   |        |</p>
</li>
</ol>
<h2>FAQ</h2>
<h3>1. What does a delegate summary result with a <code>(strict)</code> suffix mean?</h3>
<p>The <code>(strict)</code> suffix is added to reference delegates that have the same
delegate type as the test target delegate. The purpose of the suffix is to let a
user know that the performance metrics are being checked to a higher standard.
The expectation is that the test target delegate is better, or at least not
substantially worse, than the reference delegate in all metrics.</p>
<p>Please see
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/java/org/tensorflow/lite/benchmark/delegateperformance/BenchmarkResultType.java">BenchmarkResultType.java</a>
for more details.</p>