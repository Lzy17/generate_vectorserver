<h1>Builtin Ops List Generator.</h1>
<p>This directory contains a code generator to generate a pure C header for
builtin ops lists.</p>
<p>Whenever you add a new builtin op, please execute:</p>
<p><code>sh
bazel run \
  //tensorflow/lite/schema/builtin_ops_header:generate &gt; \
  tensorflow/lite/builtin_ops.h &amp;&amp;
bazel run \
  //tensorflow/lite/schema/builtin_ops_list:generate &gt; \
  tensorflow/lite/kernels/builtin_ops_list.inc</code></p>