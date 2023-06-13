<h1>TFLite Model Benchmark Tool with C++ Binary</h1>
<h2>Description</h2>
<p>A simple C++ binary to benchmark a TFLite model and its individual operators,
both on desktop machines and on Android. The binary takes a TFLite model,
generates random inputs and then repeatedly runs the model for specified number
of runs. Aggregate latency statistics are reported after running the benchmark.</p>
<p>The instructions below are for running the binary on Desktop and Android,
for iOS please use the
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/ios">iOS benchmark app</a>.</p>
<p>An experimental Android APK wrapper for the benchmark model utility offers more
faithful execution behavior on Android (via a foreground Activity). It is
located
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android">here</a>.</p>
<h2>Parameters</h2>
<p>The binary takes the following required parameters:</p>
<ul>
<li><code>graph</code>: <code>string</code> \
    The path to the TFLite model file.</li>
</ul>
<p>and the following optional parameters:</p>
<ul>
<li><code>num_threads</code>: <code>int</code> (default=-1) \
    The number of threads to use for running TFLite interpreter. By default,
    this is set to the platform default value -1.</li>
<li><code>warmup_runs</code>: <code>int</code> (default=1) \
    The number of warmup runs to do before starting the benchmark.</li>
<li><code>num_runs</code>: <code>int</code> (default=50) \
    The number of runs. Increase this to reduce variance.</li>
<li><code>max_secs</code> : float (default=150.0) \
    The maximum number of seconds the benchmark will run before being
    terminated.</li>
<li><code>run_delay</code>: <code>float</code> (default=-1.0) \
    The delay in seconds between subsequent benchmark runs. Non-positive values
    mean use no delay.</li>
<li><code>run_frequency</code>: <code>float</code> (default=-1.0) \
    The frequency of running a benchmark run as the number of prorated runs per
    second. If the targeted rate per second cannot be reached, the benchmark
    would start the next run immediately, trying its best to catch up. If set,
    this will override the <code>run_delay</code> parameter. A non-positive value means
    there is no delay between subsequent runs.</li>
<li><code>enable_op_profiling</code>: <code>bool</code> (default=false) \
    Whether to enable per-operator profiling measurement.</li>
<li><code>max_profiling_buffer_entries</code>: <code>int</code> (default=1024) \
    The initial max number of profiling events that will be stored during each
    inference run. It is only meaningful when <code>enable_op_profiling</code> is set to
    <code>true</code>. Note, the actual value of this parameter will be adjusted if the
    model has more nodes than the specified value of this parameter. Also, when
    <code>allow_dynamic_profiling_buffer_increase</code> is set to <code>true</code>, the number of
    profiling buffer entries will be increased dynamically.</li>
<li>
<p><code>allow_dynamic_profiling_buffer_increase</code>: <code>bool</code> (default=false) \
    Whether allowing dynamic increase on the number of profiling buffer entries.
    It is only meaningful when <code>enable_op_profiling</code> is set to <code>true</code>. Note,
    allowing dynamic buffer size increase may cause more profiling overhead,
    thus it is preferred to set <code>max_profiling_buffer_entries</code> to a large-enough
    value.</p>
</li>
<li>
<p><code>profiling_output_csv_file</code>: <code>str</code> (default="") \
    File path to export profile data to as CSV. The results are printed to
    <code>stdout</code> if option is not set. Requires <code>enable_op_profiling</code> to be <code>true</code>
    and the path to include the name of the output CSV; otherwise results are
    printed to <code>stdout</code>.</p>
</li>
<li>
<p><code>print_preinvoke_state</code>: <code>bool</code> (default=false) \
    Whether to print out the TfLite interpreter internals just before calling
    tflite::Interpreter::Invoke. The internals will include allocated memory
    size of each tensor etc. Enabling this could help understand TfLite graph
    and memory usage.</p>
</li>
<li>
<p><code>print_postinvoke_state</code>: <code>bool</code> (default=false) \
    Whether to print out the TfLite interpreter internals just before benchmark
    completes (i.e. after all repeated Invoke calls complete). The internals
    will include allocated memory size of each tensor etc. Enabling this could
    help understand TfLite graph and memory usage, particularly when there are
    dynamic-shaped tensors in the graph.</p>
</li>
<li>
<p><code>report_peak_memory_footprint</code>: <code>bool</code> (default=false) \
    Whether to report the peak memory footprint by periodically checking the
    memory footprint. Internally, a separate thread will be spawned for this
    periodic check. Therefore, the performance benchmark result could be
    affected.</p>
</li>
<li>
<p><code>memory_footprint_check_interval_ms</code>: <code>int</code> (default=50) \
    The interval in millisecond between two consecutive memory footprint checks.
    This is only used when --report_peak_memory_footprint is set to true.</p>
</li>
<li>
<p><code>dry_run</code>: <code>bool</code> (default=false) \
    Whether to run the tool just with simply loading the model, allocating
    tensors etc. but without actually invoking any op kernels.</p>
</li>
<li>
<p><code>verbose</code>: <code>bool</code> (default=false) \
    Whether to log parameters whose values are not set. By default, only log
    those parameters that are set by parsing their values from the commandline
    flags.</p>
</li>
<li>
<p><code>release_dynamic_tensors</code>: <code>bool</code> (default=false) \
    Whether to configure the Interpreter to immediately release the memory of
    dynamic tensors in the graph once they are not used.</p>
</li>
<li>
<p><code>optimize_memory_for_large_tensors</code>: <code>int</code> (default=0) \
    Whether to optimize memory usage for large tensors with sacrificing latency.
    When the feature is enabled, <code>release_dynamic_tensors</code> is also enabled.</p>
</li>
</ul>
<p>This list of parameters is not exhaustive. See
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/benchmark_model.cc">here</a>
and
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/benchmark_tflite_model.cc">here</a>
for all parameters that the binary takes.</p>
<h3>Model input parameters</h3>
<p>By default, the tool will use randomized data for model inputs. The following
parameters allow users to specify customized input values to the model when
running the benchmark tool:</p>
<ul>
<li><code>input_layer</code>: <code>string</code> \
    A comma-separated list of input layer names, e.g. 'input1,input2'. Note all
    inputs of the model graph need to be specified. However, the input name
    does not need to match that encoded in the model. Additionally, the order
    of input layer names specified here is assumed to be same with that is seen
    by the Tensorflow Lite interpreter. This is a bit inconvenient but the
    <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py">visualization tool</a>
    should help to find this order.</li>
<li><code>input_layer_shape</code>: <code>string</code> \
    A colon-separated list of input layer shapes, where each shape is a
    comma-separated list, e.g. '1,30:1,10'. Similar to <code>input_layer</code>, this
    parameter also requires shapes of all inputs be specified, and the order of
    inputs be same with that is seen by the interpreter.</li>
<li><code>input_layer_value_range</code>: <code>string</code> \
    A map-like string representing value range for <em>integer</em> input layers. Each
    item is separated by ':', and the item value consists of input layer name
    and integer-only range values (both low and high are inclusive) separated by
    ',', e.g. 'input1,1,2:input2,0,254'. Note that the input layer name must
    exist in the list of names specified by <code>input_layer</code>.</li>
<li><code>input_layer_value_files</code>: <code>string</code> \
    A map-like string representing files that contain input values. Each
    item is separated by ',', and the item value consists of input layer name
    and the file path separated by ':',
    e.g. 'input1:file_path1,input2:file_path2'. In case the input layer name
    contains ':' e.g. "input:0", escape it with "::" literal,
    e.g. <code>input::0:file_path1</code>. If a input name appears in both
    <code>input_layer_value_range</code> and <code>input_layer_value_files</code>, the corresponding
    input value range specified by<code>input_layer_value_range</code> will be ignored.
    The file format is binary, and the content should be either a byte array or
    null-separated strings. Note that the input layer name must also exist in
    the list of names specified by <code>input_layer</code>.</li>
</ul>
<h3>TFLite delegate parameters</h3>
<p>The tool supports all runtime/delegate parameters introduced by
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates">the delegate registrar</a>.
The following simply lists the names of these parameters and additional notes
where applicable. For details about each parameter, please refer to
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar">this page</a>.</p>
<h4>Common parameters</h4>
<ul>
<li><code>max_delegated_partitions</code>: <code>int</code> (default=0)</li>
<li><code>min_nodes_per_partition</code>:<code>int</code> (default=0)</li>
<li><code>delegate_serialize_dir</code>: <code>str</code> (default="")</li>
<li><code>delegate_serialize_token</code>: <code>str</code> (default="")</li>
</ul>
<h4>GPU delegate</h4>
<ul>
<li><code>use_gpu</code>: <code>bool</code> (default=false)</li>
<li><code>gpu_precision_loss_allowed</code>: <code>bool</code> (default=true)</li>
<li><code>gpu_experimental_enable_quant</code>: <code>bool</code> (default=true)</li>
<li><code>gpu_inference_for_sustained_speed</code>: <code>bool</code> (default=false)</li>
<li><code>gpu_backend</code>: <code>string</code> (default="")</li>
<li><code>gpu_wait_type</code>: <code>str</code> (default="")</li>
</ul>
<h4>NNAPI delegate</h4>
<ul>
<li><code>use_nnapi</code>: <code>bool</code> (default=false) \
    Note some Android P devices will fail to use NNAPI for models in
    <code>/data/local/tmp/</code> and this benchmark tool will not correctly use NNAPI.</li>
<li><code>nnapi_execution_preference</code>: <code>str</code> (default="") \
    Should be one of: <code>fast_single_answer</code>, <code>sustained_speed</code>, <code>low_power</code>,
    <code>undefined</code>.</li>
<li><code>nnapi_execution_priority</code>: <code>str</code> (default="") \
    Note this requires Android 11+.</li>
<li><code>nnapi_accelerator_name</code>: <code>str</code> (default="") \
    Note this requires Android 10+.</li>
<li><code>disable_nnapi_cpu</code>: <code>bool</code> (default=true)</li>
<li><code>nnapi_allow_fp16</code>: <code>bool</code> (default=false)</li>
<li><code>nnapi_allow_dynamic_dimensions</code>:<code>bool</code> (default=false)</li>
<li><code>nnapi_use_burst_mode</code>:<code>bool</code> (default=false)</li>
</ul>
<h4>Hexagon delegate</h4>
<ul>
<li><code>use_hexagon</code>: <code>bool</code> (default=false)</li>
<li><code>hexagon_profiling</code>: <code>bool</code> (default=false) \
Note enabling this option will not produce profiling results outputs unless
<code>enable_op_profiling</code> is also turned on. When both parameters are set to true,
the profile of ops on hexagon DSP will be added to the profile table. Note that,
the reported data on hexagon is in cycles, not in ms like on cpu.</li>
<li><code>hexagon_lib_path</code>: <code>string</code> (default="/data/local/tmp/") \
The library path for the underlying Hexagon libraries.
This is where libhexagon_nn_skel*.so files should be.
For libhexagon_interface.so it needs to be on a path that can be loaded from
example: put it in LD_LIBRARY_PATH.</li>
</ul>
<h4>XNNPACK delegate</h4>
<ul>
<li><code>use_xnnpack</code>: <code>bool</code> (default=false) \
Note if this option is explicitly set to <code>false</code>, the TfLite runtime will use
its original CPU kernels for model execution. In other words, after enabling
the feature that the XNNPACK delegate is applied by default in TfLite runtime,
explictly setting this flag to <code>false</code> will cause the benchmark tool to disable
the feature at runtime, and to use the original non-delegated CPU execution path
for model benchmarking.</li>
</ul>
<h4>CoreML delegate</h4>
<ul>
<li><code>use_coreml</code>: <code>bool</code> (default=false)</li>
<li><code>coreml_version</code>: <code>int</code> (default=0)</li>
</ul>
<h4>External delegate</h4>
<ul>
<li><code>external_delegate_path</code>: <code>string</code> (default="")</li>
<li><code>external_delegate_options</code>: <code>string</code> (default="")</li>
</ul>
<h4>Stable delegate [Experimental]</h4>
<ul>
<li><code>stable_delegate_loader_settings</code>: <code>string</code> (default="") A path to the
    JSON-encoded delegate <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/acceleration/configuration/configuration.proto#L488"><code>TFLiteSettings</code></a> file, which is defined in <code>configuration.proto</code>.</li>
</ul>
<p>As some delegates are only available on certain platforms, when running the
benchmark tool on a particular platform, specifying <code>--help</code> will print out all
supported parameters.</p>
<h3>Use multiple delegates</h3>
<p>When multiple delegates are specified to be used in the commandline flags, the
order of delegates applied to the TfLite runtime will be same as their enabling
commandline flag is specified. For example, "--use_xnnpack=true --use_gpu=true"
means applying the XNNPACK delegate first, and then the GPU delegate secondly.
In comparison, "--use_gpu=true --use_xnnpack=true" means applying the GPU
delegate first, and then the XNNPACK delegate secondly.</p>
<h2>To build/install/run</h2>
<h3>On Android:</h3>
<p>(0) Refer to https://www.tensorflow.org/lite/guide/build_android to edit the
<code>WORKSPACE</code> to configure the android NDK/SDK.</p>
<p>(1) Build for your specific platform, e.g.:</p>
<p><code>bazel build -c opt \
  --config=android_arm64 \
  tensorflow/lite/tools/benchmark:benchmark_model</code></p>
<p>(2) Connect your phone. Push the binary to your phone with adb push
     (make the directory if required):</p>
<p><code>adb push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp</code></p>
<p>(3) Make the binary executable.</p>
<p><code>adb shell chmod +x /data/local/tmp/benchmark_model</code></p>
<p>(4) Push the compute graph that you need to test. For example:</p>
<p><code>adb push mobilenet_quant_v1_224.tflite /data/local/tmp</code></p>
<p>(5) Optionally, install Hexagon libraries on device.</p>
<p>That step is only needed when using the Hexagon delegate.</p>
<p><code>bazel build --config=android_arm64 \
  tensorflow/lite/delegates/hexagon/hexagon_nn:libhexagon_interface.so
adb push bazel-bin/tensorflow/lite/delegates/hexagon/hexagon_nn/libhexagon_interface.so /data/local/tmp
adb push libhexagon_nn_skel*.so /data/local/tmp</code></p>
<p>(6) Run the benchmark. For example:</p>
<p><code>adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --num_threads=4</code></p>
<h3>On desktop:</h3>
<p>(1) build the binary</p>
<p><code>bazel build -c opt tensorflow/lite/tools/benchmark:benchmark_model</code></p>
<p>(2) Run on your compute graph, similar to the Android case but without the need of adb shell.
For example:</p>
<p><code>bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=mobilenet_quant_v1_224.tflite \
  --num_threads=4</code></p>
<p>The MobileNet graph used as an example here may be downloaded from <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip">here</a>.</p>
<h2>Reducing variance between runs on Android.</h2>
<p>Most modern Android phones use <a href="https://en.wikipedia.org/wiki/ARM_big.LITTLE">ARM big.LITTLE</a>
architecture where some cores are more power hungry but faster than other cores.
When running benchmarks on these phones there can be significant variance
between different runs of the benchmark. One way to reduce variance between runs
is to set the <a href="https://en.wikipedia.org/wiki/Processor_affinity">CPU affinity</a>
before running the benchmark. On Android this can be done using the <code>taskset</code>
command.
E.g. for running the benchmark on big cores on Pixel 2 with a single thread one
can use the following command:</p>
<p><code>adb shell taskset f0 /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --num_threads=1</code></p>
<p>where <code>f0</code> is the affinity mask for big cores on Pixel 2.
Note: The affinity mask varies with the device.</p>
<h2>Profiling model operators</h2>
<p>The benchmark model binary also allows you to profile operators and give
execution times of each operator. To do this, pass the flag
<code>--enable_op_profiling=true</code> to <code>benchmark_model</code> during invocation, e.g.,</p>
<p><code>adb shell taskset f0 /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --enable_op_profiling=true</code></p>
<p>When enabled, the <code>benchmark_model</code> binary will produce detailed statistics for
each operation similar to those shown below:</p>
<p>```</p>
<p>============================== Run Order ==============================
                 [node type]      [start]     [first]    [avg ms]        [%]      [cdf%]      [mem KB]  [times called]  [Name]
                     CONV_2D        0.000       4.269       4.269     0.107%      0.107%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_0/Relu6]
           DEPTHWISE_CONV_2D        4.270       2.150       2.150     0.054%      0.161%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6]
                     CONV_2D        6.421       6.107       6.107     0.153%      0.314%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6]
           DEPTHWISE_CONV_2D       12.528       1.366       1.366     0.034%      0.348%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6]
                     CONV_2D       13.895       4.195       4.195     0.105%      0.454%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6]
           DEPTHWISE_CONV_2D       18.091       1.260       1.260     0.032%      0.485%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6]
                     CONV_2D       19.352       6.652       6.652     0.167%      0.652%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6]
           DEPTHWISE_CONV_2D       26.005       0.698       0.698     0.018%      0.670%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6]
                     CONV_2D       26.703       3.344       3.344     0.084%      0.754%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6]
           DEPTHWISE_CONV_2D       30.047       0.646       0.646     0.016%      0.770%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6]
                     CONV_2D       30.694       5.800       5.800     0.145%      0.915%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6]
           DEPTHWISE_CONV_2D       36.495       0.331       0.331     0.008%      0.924%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6]
                     CONV_2D       36.826       2.838       2.838     0.071%      0.995%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6]
           DEPTHWISE_CONV_2D       39.665       0.439       0.439     0.011%      1.006%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6]
                     CONV_2D       40.105       5.293       5.293     0.133%      1.139%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6]
           DEPTHWISE_CONV_2D       45.399       0.352       0.352     0.009%      1.147%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6]
                     CONV_2D       45.752       5.322       5.322     0.133%      1.281%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6]
           DEPTHWISE_CONV_2D       51.075       0.357       0.357     0.009%      1.290%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6]
                     CONV_2D       51.432       5.693       5.693     0.143%      1.433%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6]
           DEPTHWISE_CONV_2D       57.126       0.366       0.366     0.009%      1.442%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6]
                     CONV_2D       57.493       5.472       5.472     0.137%      1.579%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6]
           DEPTHWISE_CONV_2D       62.966       0.364       0.364     0.009%      1.588%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6]
                     CONV_2D       63.330       5.404       5.404     0.136%      1.724%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6]
           DEPTHWISE_CONV_2D       68.735       0.155       0.155     0.004%      1.728%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6]
                     CONV_2D       68.891       2.970       2.970     0.074%      1.802%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6]
           DEPTHWISE_CONV_2D       71.862       0.206       0.206     0.005%      1.807%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6]
                     CONV_2D       72.069       5.888       5.888     0.148%      1.955%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6]
             AVERAGE_POOL_2D       77.958       0.036       0.036     0.001%      1.956%         0.000          0   [MobilenetV1/Logits/AvgPool_1a/AvgPool]
                     CONV_2D       77.994       1.445       1.445     0.036%      1.992%         0.000          0   [MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd]
                     RESHAPE       79.440       0.002       0.002     0.000%      1.992%         0.000          0   [MobilenetV1/Predictions/Reshape]
                     SOFTMAX       79.443       0.029       0.029     0.001%      1.993%         0.000          0   [MobilenetV1/Predictions/Softmax]</p>
<p>============================== Top by Computation Time ==============================
                 [node type]      [start]     [first]    [avg ms]        [%]      [cdf%]      [mem KB]  [times called]  [Name]
                     CONV_2D       19.352       6.652       6.652     0.167%      0.167%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6]
                     CONV_2D        6.421       6.107       6.107     0.153%      0.320%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6]
                     CONV_2D       72.069       5.888       5.888     0.148%      0.468%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6]
                     CONV_2D       30.694       5.800       5.800     0.145%      0.613%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6]
                     CONV_2D       51.432       5.693       5.693     0.143%      0.756%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6]
                     CONV_2D       57.493       5.472       5.472     0.137%      0.893%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6]
                     CONV_2D       63.330       5.404       5.404     0.136%      1.029%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6]
                     CONV_2D       45.752       5.322       5.322     0.133%      1.162%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6]
                     CONV_2D       40.105       5.293       5.293     0.133%      1.295%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6]
                     CONV_2D        0.000       4.269       4.269     0.107%      1.402%         0.000          0   [MobilenetV1/MobilenetV1/Conv2d_0/Relu6]</p>
<p>Number of nodes executed: 31
============================== Summary by node type ==============================
                 [Node type]      [count]     [avg ms]      [avg %]     [cdf %]   [mem KB]  [times called]
                     CONV_2D           15        1.406      89.270%     89.270%      0.000          0
           DEPTHWISE_CONV_2D           13        0.169      10.730%    100.000%      0.000          0
                     SOFTMAX            1        0.000       0.000%    100.000%      0.000          0
                     RESHAPE            1        0.000       0.000%    100.000%      0.000          0
             AVERAGE_POOL_2D            1        0.000       0.000%    100.000%      0.000          0</p>
<p>Timings (microseconds): count=50 first=79449 curr=81350 min=77385 max=88213 avg=79732 std=1929
Memory (bytes): count=0
31 nodes observed</p>
<p>Average inference timings in us: Warmup: 83235, Init: 38467, Inference: 79760.9
```</p>
<h2>Benchmark multiple performance options in a single run</h2>
<p>A convenient and simple C++ binary is also provided to benchmark multiple
performance options in a single run. This binary is built based on the
aforementioned benchmark tool that could only benchmark a single performance
option at a time. They share the same build/install/run process, but the BUILD
target name of this binary is <code>benchmark_model_performance_options</code> and it takes
some additional parameters as detailed below.</p>
<h3>Additional Parameters</h3>
<ul>
<li><code>perf_options_list</code>: <code>string</code> (default='all') \
    A comma-separated list of TFLite performance options to benchmark.</li>
<li><code>option_benchmark_run_delay</code>: <code>float</code> (default=-1.0) \
    The delay between two consecutive runs of benchmarking performance options
    in seconds.</li>
<li><code>random_shuffle_benchmark_runs</code>: <code>bool</code> (default=true) \
    Whether to perform all benchmark runs, each of which has different
    performance options, in a random order.</li>
</ul>
<h2>Build the benchmark tool with Tensorflow ops support</h2>
<p>You can build the benchmark tool with <a href="https://www.tensorflow.org/lite/guide/ops_select">Tensorflow operators support</a>.</p>
<h3>How to build</h3>
<p>To build the tool, you need to use 'benchmark_model_plus_flex' target with
'--config=monolithic' option.</p>
<p><code>bazel build -c opt \
  --config=monolithic \
  tensorflow/lite/tools/benchmark:benchmark_model_plus_flex</code></p>
<h3>How to benchmark tflite model with Tensorflow ops</h3>
<p>Tensorflow ops support just works the benchmark tool is built with Tensorflow
ops support. It doesn't require any additional option to use it.</p>
<p><code>bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_plus_flex \
  --graph=model_converted_with_TF_ops.tflite \</code></p>