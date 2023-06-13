<p>This directory contains the "core" part of the TensorFlow Lite runtime library.
The header files in this <code>tensorflow/lite/core/</code> directory fall into several
categories.</p>
<ol>
<li>
<p>Public API headers, in the <code>api</code> subdirectory <code>tensorflow/lite/core/api/</code></p>
<p>These are in addition to the other public API headers in <code>tensorflow/lite/</code>.</p>
<p>For example:
- <code>tensorflow/lite/core/api/error_reporter.h</code>
- <code>tensorflow/lite/core/api/op_resolver.h</code></p>
</li>
<li>
<p>Private headers that define public API types and functions.
    These headers are each <code>#include</code>d from a corresponding public "shim" header
    in <code>tensorflow/lite/</code> that forwards to the private header.</p>
<p>For example:
- <code>tensorflow/lite/core/interpreter.h</code> is a private header file that is
  included from the public "shim" header file <code>tensorflow/lite/interpeter.h</code>.</p>
<p>These private header files should be used as follows: <code>#include</code>s from <code>.cc</code>
files in TF Lite itself that are <em>implementing</em> the TF Lite APIs should
include the "core" TF Lite API headers.  <code>#include</code>s from files that are
just <em>using</em> the regular TF Lite APIs should include the regular public
headers.</p>
</li>
<li>
<p>The header file <code>tensorflow/lite/core/subgraph.h</code>. This contains
    some experimental APIs.</p>
</li>
</ol>