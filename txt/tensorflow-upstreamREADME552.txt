<h1>TFLite iOS benchmark app.</h1>
<h2>Description</h2>
<p>An iOS app to benchmark TFLite models.</p>
<p>The app reads benchmark parameters from a JSON file named
<code>benchmark_params.json</code> in its <code>benchmark_data</code> directory. Any downloaded models
for benchmarking should also be placed in <code>benchmark_data</code> directory.</p>
<p>The JSON file specifies the name of the model file and other benchmarking
parameters like inputs to the model, type of inputs, number of iterations,
number of threads. The default values in the JSON file are for the
Mobilenet_1.0_224 model (<a href="https://arxiv.org/pdf/1704.04861.pdf">paper</a>,
<a href="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz">tflite&amp;pb</a>).</p>
<h2>Building / running the app</h2>
<ul>
<li>
<p>Follow the <a href="https://tensorflow.org/lite/guide/build_ios">iOS build instructions</a> to configure the Bazel
    workspace and <code>.bazelrc</code> file correctly.</p>
</li>
<li>
<p>Run <code>build_benchmark_framework.sh</code> script to build the benchmark framework.
    This script will build the benchmark framework targeting iOS arm64 and put
    it under <code>TFLiteBenchmark/TFLiteBenchmark/Frameworks</code> directory.</p>
</li>
<li>
<p>If you want more detailed profiling, run the build script with <code>-p</code> option:
    <code>build_benchmark_framework.sh -p</code>.</p>
</li>
<li>
<p>Modify <code>benchmark_params.json</code> change the <code>input_layer</code>, <code>input_layer_shape</code>
    and other benchmark parameters.</p>
</li>
<li>
<p>Change <code>Build Phases -&gt; Copy Bundle Resources</code> and add the model file to the
    resources that need to be copied.</p>
</li>
<li>
<p>Ensure that <code>Build Phases -&gt; Link Binary With Library</code> contains the
    <code>Accelerate framework</code> and <code>TensorFlowLiteBenchmarkC.framework</code>.</p>
</li>
<li>
<p>Now try running the app. The app has a single button that runs the benchmark
    on the model and displays results in a text view below. You can also see the
    console output section in your Xcode to see more detailed benchmark
    information.</p>
</li>
</ul>