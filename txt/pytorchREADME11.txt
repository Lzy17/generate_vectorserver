<p>All files living in this directory are written with the assumption that MKL is available,
which means that these code are not guarded by <code>#if AT_MKL_ENABLED()</code>. Therefore, whenever
you need to use definitions from here, please guard the <code>#include&lt;ATen/mkl/*.h&gt;</code> and
definition usages with <code>#if AT_MKL_ENABLED()</code> macro, e.g. <a href="native/mkl/SpectralOps.cpp">SpectralOps.cpp</a>.</p>