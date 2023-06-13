<h1>TF Lite Android Image Classifier App Example</h1>
<p>A simple Android example that demonstrates image classification using the camera.</p>
<h2>Building in Android Studio with TensorFlow Lite AAR from MavenCentral.</h2>
<p>The build.gradle is configured to use TensorFlow Lite's nightly build.</p>
<p>If you see a build error related to compatibility with Tensorflow Lite's Java API (example: method X is
undefined for type Interpreter), there has likely been a backwards compatible
change to the API. You will need to pull new app code that's compatible with the
nightly build and may need to first wait a few days for our external and internal
code to merge.</p>
<h2>Building from Source with Bazel</h2>
<ol>
<li>
<p>Follow the <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#bazel">Bazel steps for the TF Demo App</a>:</p>
</li>
<li>
<p><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#install-bazel-and-android-prerequisites">Install Bazel and Android Prerequisites</a>.
     It's easiest with Android Studio.</p>
<ul>
<li>You'll need at least SDK version 23.</li>
<li>Make sure to install the latest version of Bazel. Some distributions
    ship with Bazel 0.5.4, which is too old.</li>
<li>Bazel requires Android Build Tools <code>28.0.0</code> or higher.</li>
<li>You also need to install the Android Support Repository, available
    through Android Studio under <code>Android SDK Manager -&gt; SDK Tools -&gt;
    Android Support Repository</code>.</li>
</ul>
</li>
<li>
<p><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#edit-workspace">Edit your <code>WORKSPACE</code></a>
     to add SDK and NDK targets.</p>
<p>NOTE: As long as you have the SDK and NDK installed, the <code>./configure</code>
 script will create these rules for you. Answer "Yes" when the script asks
 to automatically configure the <code>./WORKSPACE</code>.</p>
<ul>
<li>Make sure the <code>api_level</code> in <code>WORKSPACE</code> is set to an SDK version that
    you have installed.</li>
<li>By default, Android Studio will install the SDK to <code>~/Android/Sdk</code> and
    the NDK to <code>~/Android/Sdk/ndk-bundle</code>.</li>
</ul>
</li>
<li>
<p>Build the app with Bazel. The demo needs C++11:</p>
</li>
</ol>
<p><code>shell
  bazel build -c opt //tensorflow/lite/java/demo/app/src/main:TfLiteCameraDemo</code></p>
<ol>
<li>Install the demo on a
   <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#install">debug-enabled device</a>:</li>
</ol>
<p><code>shell
  adb install bazel-bin/tensorflow/lite/java/demo/app/src/main/TfLiteCameraDemo.apk</code></p>