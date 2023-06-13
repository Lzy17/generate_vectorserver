<h2>Image Classification evaluation based on ILSVRC 2012 task</h2>
<p>This binary evaluates the following parameters of TFLite models trained for the
<a href="http://www.image-net.org/challenges/LSVRC/2012/">ILSVRC 2012 image classification task</a>:</p>
<ul>
<li>Native pre-processing latency</li>
<li>Inference latency</li>
<li>Top-K (1 to 10) accuracy values</li>
</ul>
<p>The binary takes the path to validation images and labels as inputs, along with
the model and inference-specific parameters such as delegate and number of
threads. It outputs the metrics to std-out as follows:</p>
<p><code>Num evaluation runs: 300 # Total images evaluated
Preprocessing latency: avg=13772.5(us), std_dev=0(us)
Inference latency: avg=76578.4(us), std_dev=600(us)
Top-1 Accuracy: 0.733333
Top-2 Accuracy: 0.826667
Top-3 Accuracy: 0.856667
Top-4 Accuracy: 0.87
Top-5 Accuracy: 0.89
Top-6 Accuracy: 0.903333
Top-7 Accuracy: 0.906667
Top-8 Accuracy: 0.913333
Top-9 Accuracy: 0.92
Top-10 Accuracy: 0.923333</code></p>
<p>To run the binary download the ILSVRC 2012 devkit
<a href="#downloading-ilsvrc">see instructions</a> and run the
<a href="#ground-truth-label-generation"><code>generate_validation_ground_truth</code> script</a> to
generate the ground truth labels.</p>
<h2>Parameters</h2>
<p>The binary takes the following parameters:</p>
<ul>
<li>
<p><code>model_file</code> : <code>string</code> \
    Path to the TFlite model file.</p>
</li>
<li>
<p><code>ground_truth_images_path</code>: <code>string</code> \
    The path to the directory containing ground truth images.</p>
</li>
<li>
<p><code>ground_truth_labels</code>: <code>string</code> \
    Path to ground truth labels file. This file should contain the same number
    of labels as the number images in the ground truth directory. The labels are
    assumed to be in the same order as the sorted filename of images. See
    <a href="#ground-truth-label-generation">ground truth label generation</a> section for
    more information about how to generate labels for images.</p>
</li>
<li>
<p><code>model_output_labels</code>: <code>string</code> \
    Path to the file containing labels, that is used to interpret the output of
    the model. E.g. in case of mobilenets, this is the path to
    <code>mobilenet_labels.txt</code> where each label is in the same order as the output
    1001 dimension tensor.</p>
</li>
</ul>
<p>and the following optional parameters:</p>
<ul>
<li>
<p><code>denylist_file_path</code>: <code>string</code> \
    Path to denylist file. This file contains the indices of images that are
    denylisted for evaluation. 1762 images are denylisted in ILSVRC dataset.
    For details please refer to readme.txt of ILSVRC2014 devkit.</p>
</li>
<li>
<p><code>num_images</code>: <code>int</code> (default=0) \
    The number of images to process, if 0, all images in the directory are
    processed otherwise only num_images will be processed.</p>
</li>
<li>
<p><code>num_threads</code>: <code>int</code> (default=4) \
    The number of threads to use for evaluation. Note: This does not change the
    number of TFLite Interpreter threads, but shards the dataset to speed up
    evaluation.</p>
</li>
<li>
<p><code>output_file_path</code>: <code>string</code> \
    The final metrics are dumped into <code>output_file_path</code> as a string-serialized
    instance of <code>tflite::evaluation::EvaluationStageMetrics</code>.</p>
</li>
</ul>
<p>The following optional parameters can be used to modify the inference runtime:</p>
<ul>
<li>
<p><code>num_interpreter_threads</code>: <code>int</code> (default=1) \
    This modifies the number of threads used by the TFLite Interpreter for
    inference.</p>
</li>
<li>
<p><code>delegate</code>: <code>string</code> \
    If provided, tries to use the specified delegate for accuracy evaluation.
    Valid values: "nnapi", "gpu", "hexagon".</p>
<p>NOTE: Please refer to the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/hexagon_delegate.md">Hexagon delegate documentation</a>
for instructions on how to set it up for the Hexagon delegate. The tool
assumes that <code>libhexagon_interface.so</code> and Qualcomm libraries lie in
<code>/data/local/tmp</code>.</p>
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
<h2>Downloading ILSVRC</h2>
<p>In order to use this tool to run evaluation on the full 50K ImageNet dataset,
download the data set from http://image-net.org/request.</p>
<h2>Ground truth label generation</h2>
<p>The ILSVRC 2012 devkit <code>validation_ground_truth.txt</code> contains IDs that
correspond to synset of the image. The accuracy binary however expects the
ground truth labels to contain the actual name of category instead of synset
ids. A conversion script has been provided to convert the validation ground
truth to category labels. The <code>validation_ground_truth.txt</code> can be converted by
the following steps:</p>
<p>```
ILSVRC_2012_DEVKIT_DIR=[set to path to ILSVRC 2012 devkit]
VALIDATION_LABELS=[set to  path to output]</p>
<p>python tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/generate_validation_labels.py \
--ilsvrc_devkit_dir=${ILSVRC_2012_DEVKIT_DIR} \
--validation_labels_output=${VALIDATION_LABELS}
```</p>
<h2>Running the binary</h2>
<h3>On Android</h3>
<p>(0) Refer to
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
for configuring NDK and SDK.</p>
<p>(1) Build using the following command:</p>
<p><code>bazel build -c opt \
  --config=android_arm64 \
  --cxxopt='--std=c++17' \
  //tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval</code></p>
<p>(2) Connect your phone. Push the binary to your phone with adb push (make the
directory if required):</p>
<p><code>adb push bazel-bin/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/run_eval /data/local/tmp</code></p>
<p>(3) Make the binary executable.</p>
<p><code>adb shell chmod +x /data/local/tmp/run_eval</code></p>
<p>(4) Push the TFLite model that you need to test. For example:</p>
<p><code>adb push mobilenet_quant_v1_224.tflite /data/local/tmp</code></p>
<p>(5) Push the imagenet images to device, make sure device has sufficient storage
available before pushing the dataset:</p>
<p><code>adb shell mkdir /data/local/tmp/ilsvrc_images &amp;&amp; \
adb push ${IMAGENET_IMAGES_DIR} /data/local/tmp/ilsvrc_images</code></p>
<p>(6) Push the generated validation ground labels to device.</p>
<p><code>adb push ${VALIDATION_LABELS} /data/local/tmp/ilsvrc_validation_labels.txt</code></p>
<p>(7) Push the model labels text file to device.</p>
<p><code>adb push ${MODEL_LABELS_TXT} /data/local/tmp/model_output_labels.txt</code></p>
<p>(8) Run the binary.</p>
<p><code>adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --ground_truth_images_path=/data/local/tmp/ilsvrc_images \
  --ground_truth_labels=/data/local/tmp/ilsvrc_validation_labels.txt \
  --model_output_labels=/data/local/tmp/model_output_labels.txt \
  --output_file_path=/data/local/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images.</code></p>
<h3>On Desktop</h3>
<p>(1) Build and run using the following command:</p>
<p><code>bazel run -c opt \
  -- \
  //tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval \
  --model_file=mobilenet_quant_v1_224.tflite \
  --ground_truth_images_path=${IMAGENET_IMAGES_DIR} \
  --ground_truth_labels=${VALIDATION_LABELS} \
  --model_output_labels=${MODEL_LABELS_TXT} \
  --output_file_path=/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images.</code></p>