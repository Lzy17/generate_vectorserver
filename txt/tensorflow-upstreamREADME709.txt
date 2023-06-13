<h1>The new MLIR based TensorFlow to TensorFlow Lite converter</h1>
<p>This directory contains:</p>
<ol>
<li><a href="https://github.com/llvm/llvm-project/tree/main/mlir">MLIR</a> dialects,
    transformation passes and utilities for TensorFlow Lite.</li>
</ol>
<h2>API:</h2>
<p>The API for converting TensorFlow models to TensorFlow Lite will be through
<code>tf.lite.TFLiteConverter</code>. All the conversion code is open sourced, and
the API will be integrated soon.</p>
<h3>The conversion process from TensorFlow to TensorFlow Lite includes the</h3>
<p>following major passes:</p>
<ul>
<li>Import from GraphDef, in .pb or .pbtxt  format, into MLIR.</li>
<li>Raise to Control-flow-graph. Converts TF Control Flow dialect to TF dialect.</li>
<li>The Canonicalization pass iteratively applies canonicalization
transformations in a greedy way until no further changes occur.
Canonicalization includes constant folding.</li>
<li>The Legalize pass converts TensorFlow operations to TensorFlow Lite
ones. The operations that cannot be mapped to TensorFlow Lite dialect
are left as TensorFlow operations. Unsupported op handling follows the
proposed TFLite mechanism.</li>
<li>Optimizations are performed in both the TF &amp; TFLite dialect; aiming for small
size and high performance (among the core value proposition of
TensorFlow Lite models).</li>
<li>The Export pass writes out TensorFlow Lite FlatBuffer format. This pass
operates on MLIR TensorFlow Lite dialect and is simple/direct translation.</li>
</ul>
<p>See
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/tf_tfl_passes.cc
for the full list of MLIR passes for conversion from TensorFlow to TensorFlow
Lite.</p>