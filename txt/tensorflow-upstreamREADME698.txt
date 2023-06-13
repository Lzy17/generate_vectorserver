<h1>MLIR dialects and utilities for TensorFlow, TensorFlow Lite and XLA.</h1>
<p>This module contains the MLIR
(<a href="https://mlir.llvm.org">Multi-Level Intermediate Representation</a>)
dialects and utilities for</p>
<ol>
<li>TensorFlow</li>
<li>XLA</li>
<li>TF Lite</li>
</ol>
<p>See <a href="https://mlir.llvm.org">MLIR's website</a> for complete documentation.</p>
<h2>Getting started</h2>
<p>Building dialects and utilities here follow the standard approach using
<code>bazel</code> as the rest of TensorFlow.</p>
<h3>Using local LLVM repo</h3>
<p>To develop across MLIR core and TensorFlow, it is useful to override the repo to
use a local version instead of fetching from head. This can be achieved by
setting up your local repository for Bazel build. For this you will need to
create bazel workspace and build files:</p>
<p><code>sh
LLVM_SRC=... # this the path to the LLVM local source directory you intend to use.
touch ${LLVM_SRC}/BUILD.bazel ${LLVM_SRC}/WORKSPACE</code></p>
<p>You can then use this overlay to build TensorFlow:</p>
<p><code>bazel build --override_repository="llvm-raw=${LLVM_SRC}" \
  -c opt tensorflow/compiler/mlir:tf-opt</code></p>