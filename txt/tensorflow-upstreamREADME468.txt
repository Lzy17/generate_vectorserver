<h1>Android TensorFlow support</h1>
<p>This directory defines components (a native <code>.so</code> library and a Java JAR)
geared towards supporting TensorFlow on Android. This includes:</p>
<ul>
<li>The <a href="../../java/README.md">TensorFlow Java API</a></li>
<li>A <code>TensorFlowInferenceInterface</code> class that provides a smaller API
  surface suitable for inference and summarizing performance of model execution.</li>
</ul>
<p>For example usage, see <a href="../../examples/android/src/org/tensorflow/demo/TensorFlowImageClassifier.java">TensorFlowImageClassifier.java</a>
in the <a href="../../examples/android">TensorFlow Android Demo</a>.</p>
<p>For prebuilt libraries, see the
<a href="https://ci.tensorflow.org/view/Nightly/job/nightly-android/">nightly Android build artifacts</a>
page for a recent build.</p>
<p>The TensorFlow Inference Interface is also available as a
<a href="https://bintray.com/google/tensorflow/tensorflow">JCenter package</a>
(see the tensorflow-android directory) and can be included quite simply in your
android project with a couple of lines in the project's <code>build.gradle</code> file:</p>
<p>```
allprojects {
    repositories {
        jcenter()
    }
}</p>
<p>dependencies {
    compile 'org.tensorflow:tensorflow-android:+'
}
```</p>
<p>This will tell Gradle to use the
<a href="https://bintray.com/google/tensorflow/tensorflow/_latestVersion">latest version</a>
of the TensorFlow AAR that has been released to
<a href="https://jcenter.bintray.com/org/tensorflow/tensorflow-android/">JCenter</a>.
You may replace the <code>+</code> with an explicit version label if you wish to
use a specific release of TensorFlow in your app.</p>
<p>To build the libraries yourself (if, for example, you want to support custom
TensorFlow operators), pick your preferred approach below:</p>
<h3>Bazel</h3>
<p>First follow the Bazel setup instructions described in
<a href="../../examples/android/README.md">tensorflow/examples/android/README.md</a></p>
<p>Then, to build the native TF library:</p>
<p><code>bazel build -c opt //tensorflow/tools/android/inference_interface:libtensorflow_inference.so \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cxxopt=-std=c++11 \
   --cpu=armeabi-v7a</code></p>
<p>Replacing <code>armeabi-v7a</code> with your desired target architecture.</p>
<p>The library will be located at:</p>
<p><code>bazel-bin/tensorflow/tools/android/inference_interface/libtensorflow_inference.so</code></p>
<p>To build the Java counterpart:</p>
<p><code>bazel build //tensorflow/tools/android/inference_interface:android_tensorflow_inference_java</code></p>
<p>You will find the JAR file at:</p>
<p><code>bazel-bin/tensorflow/tools/android/inference_interface/libandroid_tensorflow_inference_java.jar</code></p>
<h2>AssetManagerFileSystem</h2>
<p>This directory also contains a TensorFlow filesystem supporting the Android
asset manager. This may be useful when writing native (C++) code that is tightly
coupled with TensorFlow. For typical usage, the library above will be
sufficient.</p>