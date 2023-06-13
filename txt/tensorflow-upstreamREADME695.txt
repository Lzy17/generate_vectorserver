<h2>3rd party XLA devices</h2>
<p>This directory is intended as a place for 3rd party XLA devices which are <em>not</em>
integrated into the public repository.</p>
<p>By adding entries to the BUILD target in this directory, a third party device
can be included as a dependency of the JIT subsystem.</p>
<p>For integration into the unit test system, see the files:</p>
<ul>
<li>tensorflow/compiler/tests/plugin.bzl</li>
<li>tensorflow/compiler/xla/tests/plugin.bzl</li>
</ul>