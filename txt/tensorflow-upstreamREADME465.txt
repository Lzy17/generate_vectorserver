<h1>TensorFlow Model Benchmark Tool</h1>
<h2>Description</h2>
<p>A simple C++ binary to benchmark a compute graph and its individual operators,
both on desktop machines and on Android.</p>
<h2>To build/install/run</h2>
<h3>On Android:</h3>
<p>(0) Refer to https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
to edit the <code>WORKSPACE</code> to configure the android NDK/SDK.</p>
<p>(1) build for your specific platform, e.g.:</p>
<p><code>bazel build -c opt \
  --crosstool_top=//external:android/crosstool \
  --cpu=armeabi-v7a \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --config monolithic \
  tensorflow/tools/benchmark:benchmark_model</code></p>
<p>(2) Connect your phone. Push the binary to your phone with adb push
     (make the directory if required):</p>
<p><code>adb push bazel-bin/tensorflow/tools/benchmark/benchmark_model /data/local/tmp</code></p>
<p>(3) Push the compute graph that you need to test. For example:</p>
<p><code>adb push tensorflow_inception_graph.pb /data/local/tmp</code></p>
<p>(4) Run the benchmark. For example:</p>
<p><code>adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/tensorflow_inception_graph.pb \
  --input_layer="input:0" \
  --input_layer_shape="1,224,224,3" \
  --input_layer_type="float" \
  --output_layer="output:0"</code></p>
<h3>On desktop:</h3>
<p>(1) build the binary</p>
<p><code>bazel build -c opt tensorflow/tools/benchmark:benchmark_model</code></p>
<p>(2) Run on your compute graph, similar to the Android case but without the need
of adb shell. For example:</p>
<p><code>bazel-bin/tensorflow/tools/benchmark/benchmark_model \
  --graph=tensorflow_inception_graph.pb \
  --input_layer="input:0" \
  --input_layer_shape="1,224,224,3" \
  --input_layer_type="float" \
  --output_layer="output:0"</code></p>
<p>The Inception graph used as an example here may be downloaded from
https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip</p>
<h2>Model downloader</h2>
<p>To download TF .pb graphs of several popular models, run:</p>
<p><code>sh
bash download_models.sh</code></p>
<h2>Comparing performance with vanilla TF</h2>
<p>We provide example scripts comparing TF-oneDNN performance with vanilla TF's
that users can modify for their own benchmarks. The scripts assume that models
are already downloaded by <code>download_models.sh</code>. To run end-to-end model
performance comparison between TF-oneDNN and vanilla TF, call</p>
<p><code>sh
bash download_models.sh  # Skip this step if models are already downloaded.
bash run_onednn_benchmarks.sh</code></p>
<p>The output is a summary table in a CSV file: results.csv. Example output:</p>
<p><code>sh
Showing runtimes in microseconds. `?` means not available.
               Model,  Batch,        Vanilla,         oneDNN,    Speedup
          bert-large,      1,              x,              y,        x/y
          bert-large,     16,            ...,            ...,        ...
          bert-large,     64,            ...,            ...,        ...          
           inception,      1,            ...,            ...,        ...
           inception,     16,            ...,            ...,        ...
           inception,     64,            ...,            ...,        ...           
                                        ⋮
        ssd-resnet34,      1,              ?,            ...,          ?
        ssd-resnet34,     16,              ?,            ...,          ?
        ssd-resnet34,     64,              ?,            ...,          ?</code></p>
<p>Vanilla TF can't run <code>ssd-resnet34</code> on CPU because it doesn't support NCHW
format.</p>