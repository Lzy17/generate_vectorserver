<h1>TensorFlow Lite C API</h1>
<p>This directory contains C APIs for TensorFlow Lite. This includes C APIs
for common types, like kernels and delegates, as well as an explicit C API
for inference.</p>
<h2>Header summary</h2>
<p>Each public C header contains types and methods for specific uses:</p>
<ul>
<li><code>common.h</code> - Contains common C enums, types and methods used throughout
    TensorFlow Lite. This includes everything from error codes, to the kernel
    and delegate APIs.</li>
<li><code>builtin_op_data.h</code> - Contains op-specific data that is used for builtin
     kernels. This should only be used when (re)implementing a builtin operator.</li>
<li><code>c_api.h</code> - Contains the TensorFlow Lite C API for inference. The
     functionality here is largely equivalent (though a strict subset of) the
     functionality provided by the C++ <code>Interpreter</code> API.</li>
<li><code>c_api_experimental.h</code> - Contains experimental C API methods for inference.
     These methods are useful and usable, but aren't yet part of the stable API.</li>
</ul>
<h2>Using the C API</h2>
<p>See the <a href="c_api.h"><code>c_api.h</code></a> header for API usage details.</p>
<h2>Building the C API</h2>
<p>A native shared library target that contains the C API for inference has been
provided. Assuming a working <a href="https://bazel.build/versions/master/docs/install.html">bazel</a>
configuration, this can be built as follows:</p>
<p><code>sh
bazel build -c opt //tensorflow/lite/c:tensorflowlite_c</code></p>
<p>and for Android (replace <code>android_arm</code> with <code>android_arm64</code> for 64-bit),
assuming you've
<a href="../g3doc/android/lite_build.md">configured your project for Android builds</a>:</p>
<p><code>sh
bazel build -c opt --cxxopt=--std=c++11 --config=android_arm \
  //tensorflow/lite/c:tensorflowlite_c</code></p>
<p>The generated shared library will be available in your
<code>bazel-bin/tensorflow/lite/c</code> directory. A target which packages the shared
library together with the necessary headers (<code>c_api.h</code>, <code>c_api_experimental.h</code>
and <code>common.h</code>) will be available soon, and will also be released as a prebuilt
archive (together with existing prebuilt packages for Android/iOS).</p>