<h1>TF Lite Experimental Unity Plugin</h1>
<p>This directory contains an experimental sample Unity (2017) Plugin, based on
the experimental TF Lite C API. The sample demonstrates running inference within
Unity by way of a C# <code>Interpreter</code> wrapper.</p>
<p>Note that the native TF Lite plugin(s) <em>must</em> be built before using the Unity
Plugin, and placed in Assets/TensorFlowLite/SDK/Plugins/. For the editor (note
that the generated shared library name and suffix are platform-dependent):</p>
<p><code>sh
bazel build -c opt //tensorflow/lite/c:tensorflowlite_c</code></p>
<p>and for Android (replace <code>android_arm</code> with <code>android_arm64</code> for 64-bit):</p>
<p><code>sh
bazel build -c opt --config=android_arm //tensorflow/lite/c:tensorflowlite_c</code></p>
<p>If you encounter issues with native plugin discovery on Mac ("Darwin")
platforms, try renaming <code>libtensorflowlite_c.dylib</code> to <code>tensorflowlite_c.bundle</code>.</p>