<h1>TFLite iOS evaluation app.</h1>
<h2>Description</h2>
<p>An iOS app to evaluate TFLite models. This is mainly for running different
evaluation tasks on iOS. Right now it only supports evaluating the inference
difference between cpu and delegates.</p>
<p>The app reads evaluation parameters from a JSON file named
<code>evaluation_params.json</code> in its <code>evaluation_data</code> directory. Any downloaded
models for evaluation should also be placed in <code>evaluation_data</code> directory.</p>
<p>The JSON file specifies the name of the model file and other evaluation
parameters like number of iterations, number of threads, delegate name. The
default values in the JSON file are for the Mobilenet_v2_1.0_224 model
(<a href="https://github.com/tensorflow/tflite-support/raw/master/tensorflow_lite_support/metadata/python/tests/testdata/image_classifier/mobilenet_v2_1.0_224.tflite">tflite&amp;pb</a>).</p>
<h2>Building / running the app</h2>
<ul>
<li>
<p>Follow the <a href="https://tensorflow.org/lite/guide/build_ios">iOS build instructions</a> to configure the Bazel
    workspace and <code>.bazelrc</code> file correctly.</p>
</li>
<li>
<p>Run <code>build_evaluation_framework.sh</code> script to build the evaluation
    framework. This script will build the evaluation framework targeting iOS
    arm64 and put it under <code>TFLiteEvaluation/TFLiteEvaluation/Frameworks</code>
    directory.</p>
</li>
<li>
<p>Update evaluation parameters in <code>evaluation_params.json</code>.</p>
</li>
<li>
<p>Change <code>Build Phases -&gt; Copy Bundle Resources</code> and add the model file to the
    resources that need to be copied.</p>
</li>
<li>
<p>Ensure that <code>Build Phases -&gt; Link Binary With Library</code> contains the
    <code>Accelerate framework</code> and <code>TensorFlowLiteInferenceDiffC.framework</code>.</p>
</li>
<li>
<p>Now try running the app. The app has a single button that runs the
    evaluation on the model and displays results in a text view below. You can
    also see the console output section in your Xcode to see more detailed
    information.</p>
</li>
</ul>