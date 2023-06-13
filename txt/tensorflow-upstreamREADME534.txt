<p>This directory contains a fake operation in order to demonstrate and test the
interfaces.</p>
<p>First test op <code>SimpleOp</code> which is an op that test various attributes and input
and output types. The other one is <code>TmplOp</code> which tests a templatized kernel.</p>
<p>The contents:</p>
<h2><code>simple_op.h|cc</code>, <code>tmpl_op.h|cc</code></h2>
<p>This is where the actual implementation of this op resides</p>
<h2><code>simple_tf_op.cc</code>, <code>tmpl_tf_op.cc</code></h2>
<p>The TF op definition.</p>
<h2><code>simple_tflite_op.h|cc</code></h2>
<p>The TFLite op definition.</p>