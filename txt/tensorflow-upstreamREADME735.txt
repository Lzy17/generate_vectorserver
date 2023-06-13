<h1>XLA Experiments</h1>
<p>This folder is intended to serve as a place to collaborate on code related to
the XLA compiler, but will not end up being a part of the compiler itself.</p>
<p>As such, the code here is not necessarily production quality, and should not be
depended on from other parts of the compiler.</p>
<p>Some examples of code appropriate for this folder are:</p>
<ul>
<li>microbenchmarks that allow us to better understand various architectures</li>
<li>scripts that help with developing specific features of the compiler, which
    might remain useful after the feature is complete (general tools should
    instead go into the xla/tools directory)</li>
<li>experimental code transformations that are not yet integrated into the
    compiler</li>
</ul>
<h2>Visibility</h2>
<p>As a result of the nature of the content in this folder, its build visibility
is intentionally kept private.</p>
<p>If you need something from here elsewhere, the recommended approach is to move
it to a more suitable and production-supported location.</p>