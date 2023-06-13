<h1>Object Detection evaluation using the 2014 COCO minival dataset.</h1>
<p>This binary evaluates the following parameters of TFLite models trained for the
<strong>bounding box-based</strong>
<a href="http://cocodataset.org/#detection-eval">COCO Object Detection</a> task:</p>
<ul>
<li>Native pre-processing latency</li>
<li>Inference latency</li>
<li>mean Average Precision (mAP) averaged across IoU thresholds from 0.5 to 0.95
    (in increments of 0.05) and all object categories.</li>
</ul>
<p>The binary takes the path to validation images and a ground truth proto file as
inputs, along with the model and inference-specific parameters such as delegate
and number of threads. It outputs the metrics to std-out as follows:</p>
<p><code>Num evaluation runs: 8059
Preprocessing latency: avg=16589.9(us), std_dev=0(us)
Inference latency: avg=85169.7(us), std_dev=505(us)
Average Precision [IOU Threshold=0.5]: 0.349581
Average Precision [IOU Threshold=0.55]: 0.330213
Average Precision [IOU Threshold=0.6]: 0.307694
Average Precision [IOU Threshold=0.65]: 0.281025
Average Precision [IOU Threshold=0.7]: 0.248507
Average Precision [IOU Threshold=0.75]: 0.210295
Average Precision [IOU Threshold=0.8]: 0.165011
Average Precision [IOU Threshold=0.85]: 0.116215
Average Precision [IOU Threshold=0.9]: 0.0507883
Average Precision [IOU Threshold=0.95]: 0.0064338
Overall mAP: 0.206576</code></p>
<p>To run the binary, please follow the
<a href="#preprocessing-the-minival-dataset">Preprocessing section</a> to prepare the data,
and then execute the commands in the
<a href="#running-the-binary">Running the binary section</a>.</p>
<h2>Parameters</h2>
<p>The binary takes the following parameters:</p>
<ul>
<li>
<p><code>model_file</code> : <code>string</code> \
    Path to the TFlite model file. It should accept images preprocessed in the
    Inception format, and the output signature should be similar to the
    <a href="https://www.tensorflow.org/lite/examples/object_detection/overview#output_signature.">SSD MobileNet model</a>:</p>
</li>
<li>
<p><code>model_output_labels</code>: <code>string</code> \
    Path to labels that correspond to output of model. E.g. in case of
    COCO-trained SSD model, this is the path to a file where each line contains
    a class detected by the model in correct order, starting from 'background'.</p>
</li>
</ul>
<p>A sample model &amp; label-list combination for COCO can be downloaded from the
TFLite
<a href="https://www.tensorflow.org/lite/guide/hosted_models#object_detection">Hosted models page</a>.</p>
<ul>
<li>
<p><code>ground_truth_images_path</code>: <code>string</code> \
    The path to the directory containing ground truth images.</p>
</li>
<li>
<p><code>ground_truth_proto</code>: <code>string</code> \
    Path to file containing tflite::evaluation::ObjectDetectionGroundTruth proto
    in text format. If left empty, mAP numbers are not provided.</p>
</li>
</ul>
<p>The above two parameters can be prepared using the <code>preprocess_coco_minival</code>
script included in this folder.</p>
<ul>
<li><code>output_file_path</code>: <code>string</code> \
    The final metrics are dumped into <code>output_file_path</code> as a string-serialized
    instance of <code>tflite::evaluation::EvaluationStageMetrics</code>.</li>
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
<h3>Debug Mode</h3>
<p>The script also supports a debug mode with the following parameter:</p>
<ul>
<li><code>debug_mode</code>: <code>boolean</code> \
    Whether to enable debug mode. Per-image predictions are written to std-out
    along with metrics.</li>
</ul>
<p>Image-wise predictions are output as follows:</p>
<h1>```</h1>
<p>Image: image_1.jpg</p>
<p>Object [0]
  Score: 0.585938
  Class-ID: 5
  Bounding Box:
    Normalized Top: 0.23103
    Normalized Bottom: 0.388524
    Normalized Left: 0.559144
    Normalized Right: 0.763928
Object [1]
  Score: 0.574219
  Class-ID: 5
  Bounding Box:
    Normalized Top: 0.269571
    Normalized Bottom: 0.373971
    Normalized Left: 0.613175
    Normalized Right: 0.760507
======================================================</p>
<p>Image: image_2.jpg
...
```</p>
<p>This mode lets you debug the output of an object detection model that isn't
necessarily trained on the COCO dataset (by leaving <code>ground_truth_proto</code> empty).
The model output signature would still need to follow the convention mentioned
above, and you we still need an output labels file.</p>
<h2>Preprocessing the minival dataset</h2>
<p>To compute mAP in a consistent and interpretable way, we utilize the same 2014
COCO 'minival' dataset that is mentioned in the
<a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md">Tensorflow detection model zoo</a>.</p>
<p>The links to download the components of the validation set are:</p>
<ul>
<li><a href="http://images.cocodataset.org/zips/val2014.zip">2014 COCO Validation Images</a></li>
<li><a href="http://images.cocodataset.org/annotations/annotations_trainval2014.zip">2014 COCO Train/Val annotations</a>:
    Out of the files from this zip, we only require <code>instances_val2014.json</code>.</li>
<li><a href="https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_minival_ids.txt">minival Image IDs</a> :
    Only applies to the 2014 validation set. You would need to copy the contents
    into a text file.</li>
</ul>
<p>Since evaluation has to be performed on-device, we first filter the above data
and extract a subset that only contains the images &amp; ground-truth bounding boxes
we need.</p>
<p>To do so, we utilize the <code>preprocess_coco_minival</code> Python binary as follows:</p>
<p>```
bazel run //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:preprocess_coco_minival -- \
  --images_folder=/path/to/val2014 \
  --instances_file=/path/to/instances_val2014.json \
  --allowlist_file=/path/to/minival_allowlist.txt \
  --output_folder=/path/to/output/folder</p>
<p>```</p>
<p>Optionally, you can specify a <code>--num_images=N</code> argument, to preprocess the first
<code>N</code> image files (based on sorted list of filenames).</p>
<p>The script generates the following within the output folder:</p>
<ul>
<li>
<p><code>images/</code>: the resulting subset of the 2014 COCO Validation images.</p>
</li>
<li>
<p><code>ground_truth.pb</code>: a <code>.pb</code> (binary-format proto) file holding
    <code>tflite::evaluation::ObjectDetectionGroundTruth</code> corresponding to image
    subset.</p>
</li>
</ul>
<h2>Running the binary</h2>
<h3>On Android</h3>
<p>(0) Refer to
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
for configuring NDK and SDK.</p>
<p>(1) Build using the following command:</p>
<p><code>bazel build -c opt \
  --config=android_arm64 \
  --cxxopt='--std=c++17' \
  //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval</code></p>
<p>(2) Connect your phone. Push the binary to your phone with adb push (make the
directory if required):</p>
<p><code>adb push bazel-bin/third_party/tensorflow/lite/tools/evaluation/tasks/coco_object_detection/run_eval /data/local/tmp</code></p>
<p>(3) Make the binary executable.</p>
<p><code>adb shell chmod +x /data/local/tmp/run_eval</code></p>
<p>(4) Push the TFLite model that you need to test:</p>
<p><code>adb push ssd_mobilenet_v1_float.tflite /data/local/tmp</code></p>
<p>(5) Push the model labels text file to device.</p>
<p><code>adb push /path/to/labelmap.txt /data/local/tmp/labelmap.txt</code></p>
<p>(6) Preprocess the dataset using the instructions given in the
<a href="#preprocessing-the-minival-dataset">Preprocessing section</a> and push the data
(folder containing images &amp; ground truth proto) to the device:</p>
<p><code>adb shell mkdir /data/local/tmp/coco_validation &amp;&amp; \
adb push /path/to/output/folder /data/local/tmp/coco_validation</code></p>
<p>(7) Run the binary.</p>
<p><code>adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/ssd_mobilenet_v1_float.tflite \
  --ground_truth_images_path=/data/local/tmp/coco_validation/images \
  --ground_truth_proto=/data/local/tmp/coco_validation/ground_truth.pb \
  --model_output_labels=/data/local/tmp/labelmap.txt \
  --output_file_path=/data/local/tmp/coco_output.txt</code></p>
<p>Optionally, you could also pass in the <code>--num_interpreter_threads</code> &amp;
<code>--delegate</code> arguments to run with different configurations.</p>
<h3>On Desktop</h3>
<p>(1) Build and run using the following command:</p>
<p><code>bazel run -c opt \
  -- \
  //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval \
  --model_file=/path/to/ssd_mobilenet_v1_float.tflite \
  --ground_truth_images_path=/path/to/images \
  --ground_truth_proto=/path/to/ground_truth.pb \
  --model_output_labels=/path/to/labelmap.txt \
  --output_file_path=/path/to/coco_output.txt</code></p>