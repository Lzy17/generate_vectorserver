<h1>TensorFlow Debugger (TFDBG)</h1>
<p>[TOC]</p>
<p>TensorFlow Debugger (TFDBG) is a specialized debugger for TensorFlow's
computation runtime. TFDBG in TensorFlow 2.x provides access to:</p>
<ul>
<li>Tensor values during <a href="https://www.tensorflow.org/guide/eager">eager</a> and
  <a href="https://www.tensorflow.org/api_docs/python/tf/Graph">graph</a> execution.</li>
<li>Structure of computation graphs</li>
<li>Source code and stack traces associated with these execution and
  graph-execution events.</li>
</ul>
<h2>How to use TFDBG?</h2>
<p>TFDBG in TensorFlow 2.x consists of a Python API that enables dumping debug data
to the file system (namely <code>tf.debugging.experimental.enable_dump_debug_info()</code>)
and a TensorBoard-based GUI that provides an interactive visualization of the
debug data (i.e., <em>TensorBoard Debugger V2 Plugin</em>).</p>
<p><code>enable_dump_debug_info()</code> offers a number of levels of tensor-value
instrumentation varying in the amount of information dumped and the incurred
performance overhead.</p>
<p>See the API documentation of
<a href="https://www.tensorflow.org/api_docs/python/tf/debugging/experimental/enable_dump_debug_info"><code>tf.debugging.experimental.enable_dump_debug_info()</code></a></p>
<p>For a detailed walkthrough of the GUI TensorBoard Debugger V2 Plugin, see
<a href="https://www.tensorflow.org/tensorboard/debugger_v2">Debugging Numerical Issues in TensorFlow Programs Using TensorBoard Debugger
V2</a>.</p>
<h2>Known issues and limitations</h2>
<ol>
<li>Using <code>tf.debugging.experimental.enable_dump_debug_info()</code> leads to
    performance penalty on your TensorFlow program. The amount of slowdown
    varied depending on whether you are using TensorFlow on CPU, GPUs, or TPUs.
    The performance penalty is the highest on TPUs, followed by GPUs, and lowest
    on CPU.</li>
<li><code>tf.debugging.experimental.enable_dump_debug_info()</code> is currently
    incompatible with
    <a href="https://www.tensorflow.org/tutorials/keras/save_and_load">model saving/loading and checkpointing</a></li>
</ol>
<h2>Legacy API for TensorFlow 1.x</h2>
<p>TensorFlow 1.x's execution paradigm is different from that of TensorFlow v2; it
is based on the deprecated
<a href="https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session"><code>tf.Session</code></a>
If you are using TensorFlow 1.x, you can use the deprecated
<code>tf_debug.LocalCLIDebugWrapperSession</code> wrapper for <code>tf.Session</code>
to inspect tensor values and other types of debug information in a
terminal-based command-line interface. For details, see
<a href="https://developers.googleblog.com/2017/02/debug-tensorflow-models-with-tfdbg.html">this blog post</a>.</p>