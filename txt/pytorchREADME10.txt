<p>All files living in this directory are written with the assumption that cuDNN is available,
which means that these code are not guarded by <code>#if AT_CUDNN_ENABLED()</code>. Therefore, whenever
you need to use definitions from here, please guard the <code>#include&lt;ATen/cudnn/*.h&gt;</code> and
definition usages with <code>#if AT_CUDNN_ENABLED()</code> macro, e.g. <a href="native/cudnn/BatchNorm.cpp">native/cudnn/BatchNorm.cpp</a>.</p>