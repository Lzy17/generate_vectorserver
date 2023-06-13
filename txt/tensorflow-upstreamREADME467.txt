<h1>TensorFlow Android Camera Demo</h1>
<p>DEPRECATED: These examples are deprecated.</p>
<p>This folder contains an example application utilizing TensorFlow for Android
devices.</p>
<h2>Description</h2>
<p>The demos in this folder are designed to give straightforward samples of using
TensorFlow in mobile applications.</p>
<p>Inference is done using the <a href="../../tools/android/inference_interface">TensorFlow Android Inference
Interface</a>, which may be built
separately if you want a standalone library to drop into your existing
application. Object tracking and efficient YUV -&gt; RGB conversion are handled by
<code>libtensorflow_demo.so</code>.</p>
<p>A device running Android 5.0 (API 21) or higher is required to run the demo due
to the use of the camera2 API, although the native libraries themselves can run
on API &gt;= 14 devices.</p>
<h2>Current samples:</h2>
<ol>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/android/test/src/org/tensorflow/demo/ClassifierActivity.java">TF Classify</a>:
        Uses the <a href="https://arxiv.org/abs/1409.4842">Google Inception</a>
        model to classify camera frames in real-time, displaying the top results
        in an overlay on the camera image.</li>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/android/test/src/org/tensorflow/demo/DetectorActivity.java">TF Detect</a>:
        Demonstrates an SSD-Mobilenet model trained using the
        <a href="https://github.com/tensorflow/models/tree/master/research/object_detection/">Tensorflow Object Detection API</a>
        introduced in <a href="https://arxiv.org/abs/1611.10012">Speed/accuracy trade-offs for modern convolutional object detectors</a> to
        localize and track objects (from 80 categories) in the camera preview
        in real-time.</li>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/android/test/src/org/tensorflow/demo/StylizeActivity.java">TF Stylize</a>:
        Uses a model based on <a href="https://arxiv.org/abs/1610.07629">A Learned Representation For Artistic
        Style</a> to restyle the camera preview
        image to that of a number of different artists.</li>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/android/test/src/org/tensorflow/demo/SpeechActivity.java">TF
    Speech</a>:
    Runs a simple speech recognition model built by the <a href="https://www.tensorflow.org/versions/master/tutorials/audio_recognition">audio training
    tutorial</a>. Listens
    for a small set of words, and highlights them in the UI when they are
    recognized.</li>
</ol>
<p><img src="sample_images/classify1.jpg" width="30%"><img src="sample_images/stylize1.jpg" width="30%"><img src="sample_images/detect1.jpg" width="30%"></p>
<h2>Prebuilt Components:</h2>
<p>The fastest path to trying the demo is to download the <a href="https://storage.googleapis.com/download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk">prebuilt demo APK</a>.</p>
<p>Also available are precompiled native libraries, and a jcenter package that you
may simply drop into your own applications. See
<a href="../../tools/android/inference_interface/README.md">tensorflow/tools/android/inference_interface/README.md</a>
for more details.</p>
<h2>Running the Demo</h2>
<p>Once the app is installed it can be started via the "TF Classify", "TF Detect",
"TF Stylize", and "TF Speech" icons, which have the orange TensorFlow logo as
their icon.</p>
<p>While running the activities, pressing the volume keys on your device will
toggle debug visualizations on/off, rendering additional info to the screen that
may be useful for development purposes.</p>
<h2>Building in Android Studio using the TensorFlow AAR from JCenter</h2>
<p>The simplest way to compile the demo app yourself, and try out changes to the
project code is to use AndroidStudio. Simply set this <code>android</code> directory as the
project root.</p>
<p>Then edit the <code>build.gradle</code> file and change the value of <code>nativeBuildSystem</code> to
<code>'none'</code> so that the project is built in the simplest way possible:</p>
<p><code>None
def nativeBuildSystem = 'none'</code></p>
<p>While this project includes full build integration for TensorFlow, this setting
disables it, and uses the TensorFlow Inference Interface package from JCenter.</p>
<p>Note: Currently, in this build mode, YUV -&gt; RGB is done using a less efficient
Java implementation, and object tracking is not available in the "TF Detect"
activity.</p>
<p>For any project that does not include custom low level TensorFlow code, this is
likely sufficient.</p>
<p>For details on how to include this JCenter package in your own project see
<a href="../../tools/android/inference_interface/README.md">tensorflow/tools/android/inference_interface/README.md</a></p>
<h2>Building the Demo with TensorFlow from Source</h2>
<p>Pick your preferred approach below. At the moment, we have full support for
Bazel, and partial support for gradle and Android Studio.</p>
<p>As a first step for all build types, clone the TensorFlow repo with:</p>
<p><code>git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git</code></p>
<p>Note that <code>--recurse-submodules</code> is necessary to prevent some issues with
protobuf compilation.</p>
<h3>Bazel</h3>
<p>NOTE: Bazel does not currently support building for Android on Windows. In the
meantime we suggest that Windows users download the
<a href="https://storage.googleapis.com/download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk">prebuilt demo APK</a>
instead.</p>
<h5>Install Bazel and Android Prerequisites</h5>
<p>Bazel is the primary build system for TensorFlow. To build with Bazel, it and
the Android NDK and SDK must be installed on your system.</p>
<ol>
<li>Install the latest version of Bazel as per the instructions <a href="https://bazel.build/versions/master/docs/install.html">on the Bazel
    website</a>.</li>
<li>The Android NDK is required to build the native (C/C++) TensorFlow code. The
    current recommended version is 14b, which may be found
    <a href="https://developer.android.com/ndk/downloads/older_releases.html#ndk-14b-downloads">here</a>.</li>
<li>The Android SDK and build tools may be obtained
    <a href="https://developer.android.com/tools/revisions/build-tools.html">here</a>, or
    alternatively as part of <a href="https://developer.android.com/studio/index.html">Android
    Studio</a>. Build tools API &gt;=
    23 is required to build the TF Android demo (though it will run on API &gt;= 21
    devices).</li>
</ol>
<h5>Edit WORKSPACE</h5>
<p>NOTE: As long as you have the SDK and NDK installed, the <code>./configure</code> script
will create these rules for you. Answer "Yes" when the script asks to
automatically configure the <code>./WORKSPACE</code>.</p>
<p>The Android entries in
<a href="../../../WORKSPACE#L19-L36"><code>&lt;workspace_root&gt;/WORKSPACE</code></a> must be uncommented
with the paths filled in appropriately depending on where you installed the NDK
and SDK. Otherwise an error such as: "The external label
'//external:android/sdk' is not bound to anything" will be reported.</p>
<p>Also edit the API levels for the SDK in WORKSPACE to the highest level you have
installed in your SDK. This must be &gt;= 23 (this is completely independent of the
API level of the demo, which is defined in AndroidManifest.xml). The NDK API
level may remain at 14.</p>
<h5>Install Model Files (optional)</h5>
<p>The TensorFlow <code>GraphDef</code>s that contain the model definitions and weights are
not packaged in the repo because of their size. They are downloaded
automatically and packaged with the APK by Bazel via a new_http_archive defined
in <code>WORKSPACE</code> during the build process, and by Gradle via
download-models.gradle.</p>
<p><strong>Optional</strong>: If you wish to place the models in your assets manually, remove
all of the <code>model_files</code> entries from the <code>assets</code> list in <code>tensorflow_demo</code>
found in the <a href="BUILD#L92"><code>BUILD</code></a> file. Then download and extract the archives
yourself to the <code>assets</code> directory in the source tree:</p>
<p><code>bash
BASE_URL=https://storage.googleapis.com/download.tensorflow.org/models
for MODEL_ZIP in inception5h.zip ssd_mobilenet_v1_android_export.zip stylize_v1.zip
do
  curl -L ${BASE_URL}/${MODEL_ZIP} -o /tmp/${MODEL_ZIP}
  unzip /tmp/${MODEL_ZIP} -d tensorflow/tools/android/test/assets/
done</code></p>
<p>This will extract the models and their associated metadata files to the local
assets/ directory.</p>
<p>If you are using Gradle, make sure to remove download-models.gradle reference
from build.gradle after your manually download models; otherwise gradle might
download models again and overwrite your models.</p>
<h5>Build</h5>
<p>After editing your WORKSPACE file to update the SDK/NDK configuration, you may
build the APK. Run this from your workspace root:</p>
<p><code>bash
bazel build --cxxopt='--std=c++11' -c opt //tensorflow/tools/android/test:tensorflow_demo</code></p>
<h5>Install</h5>
<p>Make sure that adb debugging is enabled on your Android 5.0 (API 21) or later
device, then after building use the following command from your workspace root
to install the APK:</p>
<p><code>bash
adb install -r bazel-bin/tensorflow/tools/android/test/tensorflow_demo.apk</code></p>
<h3>Android Studio with Bazel</h3>
<p>Android Studio may be used to build the demo in conjunction with Bazel. First,
make sure that you can build with Bazel following the above directions. Then,
look at <a href="build.gradle">build.gradle</a> and make sure that the path to Bazel
matches that of your system.</p>
<p>At this point you can add the tensorflow/tools/android/test directory as a new
Android Studio project. Click through installing all the Gradle extensions it
requests, and you should be able to have Android Studio build the demo like any
other application (it will call out to Bazel to build the native code with the
NDK).</p>