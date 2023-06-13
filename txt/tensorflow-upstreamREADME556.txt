<h2>Inference Diff tool</h2>
<p><strong>NOTE: This is an experimental tool to analyze TensorFlow Lite behavior on
delegates.</strong></p>
<p>For a given model, this binary compares TensorFlow Lite execution (in terms of
latency &amp; output-value deviation) in two settings:</p>
<ul>
<li>Single-threaded CPU Inference</li>
<li>User-defined Inference</li>
</ul>
<p>To do so, the tool generates random gaussian data and passes it through two
TFLite Interpreters - one running single-threaded CPU kernels and the other
parameterized by the user's arguments.</p>
<p>It measures the latency of both, as well as the absolute difference between the
output tensors from each Interpreter, on a per-element basis.</p>
<p>The final output (logged to stdout) typically looks like this:</p>
<p><code>Num evaluation runs: 50
Reference run latency: avg=84364.2(us), std_dev=12525(us)
Test run latency: avg=7281.64(us), std_dev=2089(us)
OutputDiff[0]: avg_error=1.96277e-05, std_dev=6.95767e-06</code></p>
<p>There is one instance of <code>OutputDiff</code> for each output tensor in the model, and
the statistics in <code>OutputDiff[i]</code> correspond to the absolute difference in raw
values across all elements for the <code>i</code>th output.</p>
<h2>Parameters</h2>
<p>(In this section, 'test Interpreter' refers to the User-defined Inference
mentioned above. The reference setting is always single-threaded CPU).</p>
<p>The binary takes the following parameters:</p>
<ul>
<li><code>model_file</code> : <code>string</code> \
    Path to the TFlite model file.</li>
</ul>
<p>and the following optional parameters:</p>
<ul>
<li>
<p><code>num_runs</code>: <code>int</code> \
    How many runs to perform to compare execution in reference and test setting.
    Default: 50. The binary performs runs 3 invocations per 'run', to get more
    accurate latency numbers.</p>
</li>
<li>
<p><code>num_interpreter_threads</code>: <code>int</code> (default=1) \
    This modifies the number of threads used by the test Interpreter for
    inference.</p>
</li>
<li>
<p><code>delegate</code>: <code>string</code> \
    If provided, tries to use the specified delegate on the test Interpreter.
    Valid values: "nnapi", "gpu", "hexagon", "coreml".</p>
<p>NOTE: Please refer to the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/hexagon_delegate.md">Hexagon delegate documentation</a>
for instructions on how to set it up for the Hexagon delegate. The tool
assumes that <code>libhexagon_interface.so</code> and Qualcomm libraries lie in
<code>/data/local/tmp</code>.</p>
</li>
<li>
<p><code>output_file_path</code>: <code>string</code> \
    The final metrics are dumped into <code>output_file_path</code> as a serialized
    instance of <code>tflite::evaluation::EvaluationStageMetrics</code></p>
</li>
</ul>
<p>This script also supports runtime/delegate arguments introduced by the
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates">delegate registrar</a>.
If there is any conflict (for example, <code>num_threads</code> vs
<code>num_interpreter_threads</code> here), the parameters of this
script are given precedence.</p>
<p>When <strong>multiple delegates</strong> are specified to be used in the commandline flags
via the support of delegate registrar, the order of delegates applied to the
TfLite runtime will be same as their enabling commandline flag is specified. For
example, "--use_xnnpack=true --use_gpu=true" means applying the XNNPACK delegate
first, and then the GPU delegate secondly. In comparison,
"--use_gpu=true --use_xnnpack=true" means applying the GPU delegate first, and
then the XNNPACK delegate secondly.</p>
<p>Note, one could specify <code>--help</code> when launching the binary to see the full list
of supported arguments.</p>
<h2>Running the binary on Android</h2>
<p>(1) Build using the following command:</p>
<p><code>bazel build -c opt \
  --config=android_arm64 \
  //tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval</code></p>
<p>(2) Connect your phone. Push the binary to your phone with adb push (make the
directory if required):</p>
<p><code>adb push bazel-bin/third_party/tensorflow/lite/tools/evaluation/tasks/inference_diff/run_eval /data/local/tmp</code></p>
<p>(3) Push the TFLite model that you need to test. For example:</p>
<p><code>adb push mobilenet_v1_1.0_224.tflite /data/local/tmp</code></p>
<p>(3) Run the binary.</p>
<p><code>adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/mobilenet_v1_1.0_224.tflite \
  --delegate=gpu</code></p>
<p>(5) Pull the results.</p>
<p><code>adb pull /data/local/tmp/inference_diff.txt ~/accuracy_tool</code></p>
<h2>Running the binary on iOS</h2>
<p>Follow the instructions <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/ios/README.md">here</a>
to run the binary on iOS using the
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/ios">iOS evaluation app</a>.</p>