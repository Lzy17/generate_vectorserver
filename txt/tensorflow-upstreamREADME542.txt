<h1>OVIC Benchmarker for LPCV 2020</h1>
<p>This folder contains the SDK for track one of the
<a href="https://lpcv.ai/2020CVPR/ovic-track">Low Power Computer Vision workshop at CVPR 2020.</a></p>
<h2>Pre-requisite</h2>
<p>Follow the steps <a href="https://www.tensorflow.org/lite/demo_android">here</a> to install
Tensorflow, Bazel, and the Android NDK and SDK.</p>
<h2>Test the benchmarker:</h2>
<p>The testing utilities helps the developers (you) to make sure that your
submissions in TfLite format will be processed as expected in the competition's
benchmarking system.</p>
<p>Note: for now the tests only provides correctness checks, i.e. classifier
predicts the correct category on the test image, but no on-device latency
measurements. To test the latency measurement functionality, the tests will
print the latency running on a desktop computer, which is not indicative of the
on-device run-time. We are releasing an benchmarker Apk that would allow
developers to measure latency on their own devices.</p>
<h3>Obtain the sample models</h3>
<p>The test data (models and images) should be downloaded automatically for you by
Bazel. In case they are not, you can manually install them as below.</p>
<p>Note: all commands should be called from your tensorflow installation folder
(under this folder you should find <code>tensorflow/lite</code>).</p>
<ul>
<li>Download the
    <a href="https://storage.googleapis.com/download.tensorflow.org/data/ovic_2019_04_30.zip">testdata package</a>:</li>
</ul>
<p><code>sh
curl -L https://storage.googleapis.com/download.tensorflow.org/data/ovic_2019_04_30.zip -o /tmp/ovic.zip</code></p>
<ul>
<li>Unzip the package into the testdata folder:</li>
</ul>
<p><code>sh
unzip -j /tmp/ovic.zip -d tensorflow/lite/java/ovic/src/testdata/</code></p>
<h3>Run tests</h3>
<p>You can run test with Bazel as below. This helps to ensure that the installation
is correct.</p>
<p>```sh
bazel test //tensorflow/lite/java/ovic:OvicClassifierTest --cxxopt=-Wno-all --test_output=all</p>
<p>bazel test //tensorflow/lite/java/ovic:OvicDetectorTest --cxxopt=-Wno-all --test_output=all
```</p>
<h3>Test your submissions</h3>
<p>Once you have a submission that follows the instructions from the
<a href="https://lpcv.ai/2020CVPR/ovic-track">competition site</a>, you can verify it in
two ways:</p>
<h4>Validate using randomly generated images</h4>
<p>You can call the validator binary below to verify that your model fits the
format requirements. This often helps you to catch size mismatches (e.g. output
for classification should be [1, 1001] instead of [1,1,1,1001]). Let say the
submission file is located at <code>/path/to/my_model.lite</code>, then call:</p>
<p><code>sh
bazel build //tensorflow/lite/java/ovic:ovic_validator --cxxopt=-Wno-all
bazel-bin/tensorflow/lite/java/ovic/ovic_validator /path/to/my_model.lite classify</code></p>
<p>Successful validation should print the following message to terminal:</p>
<p>```
Successfully validated /path/to/my_model.lite.</p>
<p>```</p>
<p>To validate detection models, use the same command but provide "detect" as the
second argument instead of "classify".</p>
<h4>Test that the model produces sensible outcomes</h4>
<p>You can go a step further to verify that the model produces results as expected.
This helps you catch bugs during TFLite conversion (e.g. using the wrong mean
and std values).</p>
<ul>
<li>Move your submission to the testdata folder:</li>
</ul>
<p><code>sh
cp /path/to/my_model.lite tensorflow/lite/java/ovic/src/testdata/</code></p>
<ul>
<li>Resize the test image to the resolutions that are expected by your
    submission:</li>
</ul>
<p>The test images can be found at
<code>tensorflow/lite/java/ovic/src/testdata/test_image_*.jpg</code>. You may reuse these
images if your image resolutions are 128x128 or 224x224.</p>
<ul>
<li>Add your model and test image to the BUILD rule at
    <code>tensorflow/lite/java/ovic/src/testdata/BUILD</code>:</li>
</ul>
<p><code>JSON
filegroup(
    name = "ovic_testdata",
    srcs = [
        "@tflite_ovic_testdata//:detect.lite",
        "@tflite_ovic_testdata//:float_model.lite",
        "@tflite_ovic_testdata//:low_res_model.lite",
        "@tflite_ovic_testdata//:quantized_model.lite",
        "@tflite_ovic_testdata//:test_image_128.jpg",
        "@tflite_ovic_testdata//:test_image_224.jpg"
        "my_model.lite",        # &lt;--- Your submission.
        "my_test_image.jpg",    # &lt;--- Your test image.
    ],
    ...</code></p>
<ul>
<li>
<p>For classification models, modify <code>OvicClassifierTest.java</code>:</p>
<ul>
<li>
<p>change <code>TEST_IMAGE_PATH</code> to <code>my_test_image.jpg</code>.</p>
</li>
<li>
<p>change either <code>FLOAT_MODEL_PATH</code> or <code>QUANTIZED_MODEL_PATH</code> to
    <code>my_model.lite</code> depending on whether your model runs inference in float
    or
    <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize">8-bit</a>.</p>
</li>
<li>
<p>change <code>TEST_IMAGE_GROUNDTRUTH</code> (ImageNet class ID) to be consistent
    with your test image.</p>
</li>
</ul>
</li>
<li>
<p>For detection models, modify <code>OvicDetectorTest.java</code>:</p>
<ul>
<li>change <code>TEST_IMAGE_PATH</code> to <code>my_test_image.jpg</code>.</li>
<li>change <code>MODEL_PATH</code> to <code>my_model.lite</code>.</li>
<li>change <code>GROUNDTRUTH</code> (COCO class ID) to be consistent with your test
    image.</li>
</ul>
</li>
</ul>
<p>Now you can run the bazel tests to catch any runtime issues with the submission.</p>
<p>Note: Please make sure that your submission passes the test. If a submission
fails to pass the test it will not be processed by the submission server.</p>
<h2>Measure on-device latency</h2>
<p>We provide two ways to measure the on-device latency of your submission. The
first is through our competition server, which is reliable and repeatable, but
is limited to a few trials per day. The second is through the benchmarker Apk,
which requires a device and may not be as accurate as the server, but has a fast
turn-around and no access limitations. We recommend that the participants use
the benchmarker apk for early development, and reserve the competition server
for evaluating promising submissions.</p>
<h3>Running the benchmarker app</h3>
<p>Make sure that you have followed instructions in
<a href="#test-your-submissions">Test your submissions</a> to add your model to the
testdata folder and to the corresponding build rules.</p>
<p>Modify <code>tensorflow/lite/java/ovic/demo/app/OvicBenchmarkerActivity.java</code>:</p>
<ul>
<li>Add your model to the benchmarker apk by changing <code>modelPath</code> and
    <code>testImagePath</code> to your submission and test image.</li>
</ul>
<p><code>if (benchmarkClassification) {
    ...
    testImagePath = "my_test_image.jpg";
    modelPath = "my_model.lite";
  } else {  // Benchmarking detection.
  ...</code></p>
<p>If you are adding a detection model, simply modify <code>modelPath</code> and
<code>testImagePath</code> in the else block above.</p>
<ul>
<li>Adjust the benchmark parameters when needed:</li>
</ul>
<p>You can change the length of each experiment, and the processor affinity below.
<code>BIG_CORE_MASK</code> is an integer whose binary encoding represents the set of used
cores. This number is phone-specific. For example, Pixel 4 has 8 cores: the 4
little cores are represented by the 4 less significant bits, and the 4 big cores
by the 4 more significant bits. Therefore a mask value of 16, or in binary
<code>00010000</code>, represents using only the first big core. The mask 32, or in binary
<code>00100000</code> uses the second big core and should deliver identical results as the
mask 16 because the big cores are interchangeable.</p>
<p><code>/** Wall time for each benchmarking experiment. */
  private static final double WALL_TIME = 3000;
  /** Maximum number of iterations in each benchmarking experiment. */
  private static final int MAX_ITERATIONS = 100;
  /** Mask for binding to a single big core. Pixel 1 (4), Pixel 4 (16). */
  private static final int BIG_CORE_MASK = 16;</code></p>
<p>Note: You'll need ROOT access to the phone to change processor affinity.</p>
<ul>
<li>Build and install the app.</li>
</ul>
<p><code>bazel build -c opt --cxxopt=-Wno-all //tensorflow/lite/java/ovic/demo/app:ovic_benchmarker_binary
adb install -r bazel-bin/tensorflow/lite/java/ovic/demo/app/ovic_benchmarker_binary.apk</code></p>
<p>Start the app and pick a task by clicking either the <code>CLF</code> button for
classification or the <code>DET</code> button for detection. The button should turn bright
green, signaling that the experiment is running. The benchmarking results will
be displayed after about the <code>WALL_TIME</code> you specified above. For example:</p>
<p><code>my_model.lite: Average latency=158.6ms after 20 runs.</code></p>
<h3>Sample latencies</h3>
<p>Note: the benchmarking results can be quite different depending on the
background processes running on the phone. A few things that help stabilize the
app's readings are placing the phone on a cooling plate, restarting the phone,
and shutting down internet access.</p>
<p>Classification Model | Pixel 1 | Pixel 2 | Pixel 4
-------------------- | :-----: | ------: | :-----:
float_model.lite     | 97      | 113     | 37
quantized_model.lite | 73      | 61      | 13
low_res_model.lite   | 3       | 3       | 1</p>
<p>Detection Model        | Pixel 2 | Pixel 4
---------------------- | :-----: | :-----:
detect.lite            | 248     | 82
quantized_detect.lite  | 59      | 17
quantized_fpnlite.lite | 96      | 29</p>
<p>All latency numbers are in milliseconds. The Pixel 1 and Pixel 2 latency numbers
are measured on <code>Oct 17 2019</code> (Github commit hash
<a href="https://github.com/tensorflow/tensorflow/commit/4b02bc0e0ff7a0bc02264bc87528253291b7c949#diff-4e94df4d2961961ba5f69bbd666e0552">I05def66f58fa8f2161522f318e00c1b520cf0606</a>)</p>
<p>The Pixel 4 latency numbers are measured on <code>Apr 14 2020</code> (Github commit hash
<a href="https://github.com/tensorflow/tensorflow/commit/4b2cb67756009dda843c6b56a8b320c8a54373e0">4b2cb67756009dda843c6b56a8b320c8a54373e0</a>).</p>
<p>Since Pixel 4 has excellent support for 8-bit quantized models, we strongly
recommend you to check out the
<a href="https://www.tensorflow.org/lite/performance/post_training_quantization">Post-Training Quantization tutorial</a>.</p>
<p>The detection models above are both single-shot models (i.e. no object proposal
generation) using TfLite's <em>fast</em> version of Non-Max-Suppression (NMS). The fast
NMS is significant faster than the regular NMS (used by the ObjectDetectionAPI
in training) at the expense of about 1% mAP for the listed models.</p>
<h3>Latency table</h3>
<p>We have compiled a latency table for common neural network operators such as
convolutions, separable convolutions, and matrix multiplications. The table of
results is available here:</p>
<ul>
<li>https://storage.cloud.google.com/ovic-data/</li>
</ul>
<p>The results were generated by creating a small network containing a single
operation, and running the op under the test harness. For more details see the
NetAdapt paper<sup>1</sup>. We plan to expand table regularly as we test with
newer OS releases and updates to Tensorflow Lite.</p>
<h3>Sample benchmarks</h3>
<p>Below are the baseline models (MobileNetV2, MnasNet, and MobileNetV3) used to
compute the reference accuracy for ImageNet classification. The naming
convention of the models are <code>[precision]_[model
class]_[resolution]_[multiplier]</code>. Pixel 2 Latency (ms) is measured on a single
Pixel 2 big core using the competition server on <code>Oct 17 2019</code>, while Pixel 4
latency (ms) is measured on a single Pixel 4 big core using the competition
server on <code>Apr 14 2020</code>. You can find these models on TFLite's
<a href="https://www.tensorflow.org/lite/guide/hosted_models#image_classification">hosted model page</a>.</p>
<p>Model                               | Pixel 2 | Pixel 4 | Top-1 Accuracy
:---------------------------------: | :-----: | :-----: | :------------:
quant_mobilenetv2_96_35             | 4       | 1       | 0.420
quant_mobilenetv2_96_50             | 5       | 1       | 0.478
quant_mobilenetv2_128_35            | 6       | 2       | 0.474
quant_mobilenetv2_128_50            | 8       | 2       | 0.546
quant_mobilenetv2_160_35            | 9       | 2       | 0.534
quant_mobilenetv2_96_75             | 8       | 2       | 0.560
quant_mobilenetv2_96_100            | 10      | 3       | 0.579
quant_mobilenetv2_160_50            | 12      | 3       | 0.583
quant_mobilenetv2_192_35            | 12      | 3       | 0.557
quant_mobilenetv2_128_75            | 13      | 3       | 0.611
quant_mobilenetv2_192_50            | 16      | 4       | 0.616
quant_mobilenetv2_128_100           | 16      | 4       | 0.629
quant_mobilenetv2_224_35            | 17      | 5       | 0.581
quant_mobilenetv2_160_75            | 20      | 5       | 0.646
float_mnasnet_96_100                | 21      | 7       | 0.625
quant_mobilenetv2_224_50            | 22      | 6       | 0.637
quant_mobilenetv2_160_100           | 25      | 6       | 0.674
quant_mobilenetv2_192_75            | 29      | 7       | 0.674
quant_mobilenetv2_192_100           | 35      | 9       | 0.695
float_mnasnet_224_50                | 35      | 12      | 0.679
quant_mobilenetv2_224_75            | 39      | 10      | 0.684
float_mnasnet_160_100               | 45      | 15      | 0.706
quant_mobilenetv2_224_100           | 48      | 12      | 0.704
float_mnasnet_224_75                | 55      | 18      | 0.718
float_mnasnet_192_100               | 62      | 20      | 0.724
float_mnasnet_224_100               | 84      | 27      | 0.742
float_mnasnet_224_130               | 126     | 40      | 0.758
float_v3-small-minimalistic_224_100 | -       | 5       | 0.620
quant_v3-small_224_100              | -       | 5       | 0.641
float_v3-small_224_75               | -       | 5       | 0.656
float_v3-small_224_100              | -       | 7       | 0.677
quant_v3-large_224_100              | -       | 12      | 0.728
float_v3-large_224_75               | -       | 15      | 0.735
float_v3-large-minimalistic_224_100 | -       | 17      | 0.722
float_v3-large_224_100              | -       | 20      | 0.753</p>
<h3>References</h3>
<ol>
<li><strong>NetAdapt: Platform-Aware Neural Network Adaptation for Mobile
    Applications</strong><br />
    Yang, Tien-Ju, Andrew Howard, Bo Chen, Xiao Zhang, Alec Go, Mark Sandler,
    Vivienne Sze, and Hartwig Adam. In Proceedings of the European Conference
    on Computer Vision (ECCV), pp. 285-300. 2018<br />
    <a href="https://arxiv.org/abs/1804.03230">[link]</a> arXiv:1804.03230, 2018.</li>
</ol>