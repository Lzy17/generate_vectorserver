<h1>TensorFlow Lite C++ image classification demo</h1>
<p>This example shows how you can load a pre-trained and converted
TensorFlow Lite model and use it to recognize objects in images.</p>
<p>Before you begin,
make sure you <a href="https://www.tensorflow.org/install">have TensorFlow installed</a>.</p>
<p>You also need to
<a href="https://docs.bazel.build/versions/master/install.html">install Bazel</a> in order
to build this example code. And be sure you have the Python <code>future</code> module
installed:</p>
<p><code>pip install future --user</code></p>
<h2>Build the example</h2>
<p>First run <code>$TENSORFLOW_ROOT/configure</code>. To build for Android, set
Android NDK or configure NDK setting in
<code>$TENSORFLOW_ROOT/WORKSPACE</code> first.</p>
<p>Build it for desktop machines (tested on Ubuntu and OS X):</p>
<p><code>bazel build -c opt //tensorflow/lite/examples/label_image:label_image</code></p>
<p>Build it for Android ARMv8:</p>
<p><code>bazel build -c opt --config=android_arm64 \
  //tensorflow/lite/examples/label_image:label_image</code></p>
<p>Build it for Android arm-v7a:</p>
<p><code>bazel build -c opt --config=android_arm \
  //tensorflow/lite/examples/label_image:label_image</code></p>
<h2>Download sample model and image</h2>
<p>You can use any compatible model, but the following MobileNet v1 model offers
a good demonstration of a model trained to recognize 1,000 different objects.</p>
<p>```</p>
<h1>Get model</h1>
<p>curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz | tar xzv -C /tmp</p>
<h1>Get labels</h1>
<p>curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz  | tar xzv -C /tmp  mobilenet_v1_1.0_224/labels.txt</p>
<p>mv /tmp/mobilenet_v1_1.0_224/labels.txt /tmp/
```</p>
<h2>Run the sample on a desktop</h2>
<p><code>bazel-bin/tensorflow/lite/examples/label_image/label_image \
  --tflite_model /tmp/mobilenet_v1_1.0_224.tflite \
  --labels /tmp/labels.txt \
  --image tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp</code></p>
<p>You should see results like this:</p>
<p><code>Loaded model /tmp/mobilenet_v1_1.0_224.tflite
resolved reporter
invoked
average time: 68.12 ms
0.860174: 653 653:military uniform
0.0481017: 907 907:Windsor tie
0.00786704: 466 466:bulletproof vest
0.00644932: 514 514:cornet, horn, trumpet, trump
0.00608029: 543 543:drumstick</code></p>
<h2>Run the sample on an Android device</h2>
<p>Prepare data on devices, e.g.,</p>
<p><code>adb push bazel-bin/tensorflow/lite/examples/label_image/label_image  /data/local/tmp
adb push /tmp/mobilenet_v1_1.0_224.tflite  /data/local/tmp
adb push tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp  /data/local/tmp
adb push /tmp/labels.txt /data/local/tmp</code></p>
<p>Run it,</p>
<p><code>adb shell "/data/local/tmp/label_image \
    -m /data/local/tmp/mobilenet_v1_1.0_224.tflite \
    -i /data/local/tmp/grace_hopper.bmp \
    -l /data/local/tmp/labels.txt"</code></p>
<p>then you should see something like the following:</p>
<p><code>Loaded model /data/local/tmp/mobilenet_v1_1.0_224.tflite
resolved reporter
INFO: Initialized TensorFlow Lite runtime.
invoked
average time: 25.03 ms
0.907071: 653 military uniform
0.0372416: 907 Windsor tie
0.00733753: 466 bulletproof vest
0.00592852: 458 bow tie
0.00414091: 514 cornet</code></p>
<p>Run the model with NNAPI delegate (<code>-a 1</code>),</p>
<p><code>adb shell "/data/local/tmp/label_image \
    -m /data/local/tmp/mobilenet_v1_1.0_224.tflite \
    -i /data/local/tmp/grace_hopper.bmp \
    -l /data/local/tmp/labels.txt -a 1 -f 1"</code></p>
<p>then you should see something like the following:</p>
<p><code>Loaded model /data/local/tmp/mobilenet_v1_1.0_224.tflite
resolved reporter
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite delegate for NNAPI. Applied NNAPI delegate.
invoked
average time:10.348 ms
0.905401: 653 military uniform
0.0379589: 907 Windsor tie
0.00735866: 466 bulletproof vest
0.00605307: 458 bow tie
0.00422573: 514 cornet</code></p>
<p>To run a model with the Hexagon Delegate, assuming we have followed the
<a href="https://www.tensorflow.org/lite/android/delegates/hexagon">Hexagon Delegate Guide</a>
and installed Hexagon libraries in <code>/data/local/tmp</code>. Run it with (<code>-j 1</code>)</p>
<p><code>adb shell \
    "/data/local/tmp/label_image \
    -m /data/local/tmp/mobilenet_v1_1.0_224_quant.tflite \
    -i /data/local/tmp/grace_hopper.bmp \
    -l /data/local/tmp/labels.txt -j 1"</code></p>
<p>then you should see something like the followings:</p>
<p><code>Loaded model /data/local/tmp/mobilenet_v1_1.0_224_quant.tflite
resolved reporter
INFO: Initialized TensorFlow Lite runtime.
loaded libcdsprpc.so
INFO: Created TensorFlow Lite delegate for Hexagon.
INFO: Hexagon delegate: 31 nodes delegated out of 31 nodes with 1 partitions.
Applied Hexagon delegate.
invoked
average time: 4.231 ms
0.639216: 458 bow tie
0.329412: 653 military uniform
0.00784314: 835 suit
0.00784314: 611 jersey
0.00392157: 514 cornet</code></p>
<p>Run the model with the XNNPACK delegate (<code>-x 1</code>),</p>
<p><code>shell
adb shell \
    "/data/local/tmp/label_image \
    -m /data/local/tmp/mobilenet_v1_1.0_224.tflite \
    -i /data/local/tmp/grace_hopper.bmp \
    -l /data/local/tmp/labels.txt -x 1"</code></p>
<p>then you should see something like the following:</p>
<p><code>Loaded model /data/local/tmp/mobilenet_v1_1.0_224.tflite
resolved reporter
INFO: Initialized TensorFlow Lite runtime. Applied XNNPACK delegate.
invoked
average time: 17.33 ms
0.90707: 653 military uniform
0.0372418: 907 Windsor tie
0.0073376: 466 bulletproof vest
0.00592857: 458 bow tie
0.00414093: 514 cornet</code></p>
<p>With <code>-h</code> or any other unsupported flags, <code>label_image</code> will list supported
options:</p>
<p><code>shell
sargo:/data/local/tmp $ ./label_image -h
./label_image: invalid option -- h
label_image
--accelerated, -a: [0|1], use Android NNAPI or not
--old_accelerated, -d: [0|1], use old Android NNAPI delegate or not
--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not
--count, -c: loop interpreter-&gt;Invoke() for certain times
--gl_backend, -g: [0|1]: use GPU Delegate on Android
--hexagon_delegate, -j: [0|1]: use Hexagon Delegate on Android
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--image, -i: image_name.bmp
--labels, -l: labels for the model
--tflite_model, -m: model_name.tflite
--profiling, -p: [0|1], profiling or not
--num_results, -r: number of results to show
--threads, -t: number of threads
--verbose, -v: [0|1] print more information
--warmup_runs, -w: number of warmup runs
--xnnpack_delegate, -x [0:1]: xnnpack delegate`</code></p>
<p>See the <code>label_image.cc</code> source code for other command line options.</p>
<p>Note that this binary also supports more runtime/delegate arguments introduced
by the <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates">delegate registrar</a>.
If there is any conflict, the arguments mentioned earlier are given precedence.
For example, you can run the binary with additional command line options
such as <code>--use_nnapi=true --nnapi_accelerator_name=google-edgetpu</code> to utilize
the EdgeTPU in a 4th-gen Pixel phone. Please be aware that the "=" in the option
should not be omitted.</p>
<p><code>adb shell \
    "/data/local/tmp/label_image \
    -m /data/local/tmp/mobilenet_v1_1.0_224_quant.tflite \
    -i /data/local/tmp/grace_hopper.bmp \
    -l /data/local/tmp/labels.txt -j 1 \
    --use_nnapi=true --nnapi_accelerator_name=google-edgetpu"</code></p>