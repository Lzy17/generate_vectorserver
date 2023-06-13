<p>This directory contains build macros such as <code>cc_library_with_tflite</code>,
<code>java_library_with_tflite</code>, etc.</p>
<p><code>cc_library_with_tflite</code> generates a <code>cc_library</code> target by default.
The target will not use TF Lite in Play Services.</p>
<p>The intent is that the build macros in this directory could be modified to
optionally redirect to a different implementation of TF Lite C and C++ APIs
(for example, one built into the underlying operating system platform).</p>