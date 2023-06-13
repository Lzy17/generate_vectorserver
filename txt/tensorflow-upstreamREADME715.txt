<h1>MLIR-HLO: A Standalone "HLO" MLIR-based Compiler</h1>
<p>The code here exists in two places:</p>
<ul>
<li>https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla/mlir_hlo;
    this is the canonical location and where contributions should be made using
    GitHub pull-requests.</li>
<li>https://github.com/tensorflow/mlir-hlo; this is a standalone repository with
    a view to the same code to allow other projects to use this without
    depending on the entire TF monorepo.</li>
</ul>
<p>This implements a self-contained compiler for a linear algebra set of operations
inspired by XLA
<a href="https://www.tensorflow.org/xla/architecture#how_does_xla_work">HLO IR</a> using
MLIR components. It is designed to provide an end-to-end flow independent of
TensorFlow and XLA, but usable inside of these projects.</p>
<p>Coding practice and conventions in this repository follow the
<a href="https://mlir.llvm.org/getting_started/DeveloperGuide/">MLIR Developer Guide</a> in
this repo as part of the intent to act as an incubator for technology to
upstream.</p>
<h2>QuickStart: building and testing</h2>
<p>These instructions work on Linux, you may have to adjust for your platform.</p>
<p>To build the code in this repository, you need a clone of the LLVM/MLIR git
repository:</p>
<pre><code>$ git clone https://github.com/llvm/llvm-project.git
</code></pre>
<p>You need to make sure you have the right commit checked out in the LLVM
repository (you need to do this every time you pull from this repo):</p>
<pre><code>$ (cd llvm-project &amp;&amp; git checkout $(cat ../build_tools/llvm_version.txt))
</code></pre>
<p>We provide a script to configure and build LLVM/MLIR:</p>
<pre><code>$ build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build
</code></pre>
<p>Again this is something to do every time you pull from this repository and the
LLVM revision changes.</p>
<p>Finally you can build and test this repository:</p>
<pre><code>$ mkdir build &amp;&amp; cd build
$ cmake .. -GNinja \
   -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=On \
   -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir
$ ninja check-mlir-hlo
</code></pre>
<h2>Overview</h2>
<p>MLIR-HLO aims to provide an end-to-end compiler for CPU and GPU, as well as
building reusable blocks for other accelerators. This is heavily inspired by the
success of XLA.</p>
<p><a href="https://www.tensorflow.org/xla/">XLA</a> (Accelerated Linear Algebra) is a
domain-specific compiler framework and execution environment for linear algebra,
which powers code-generation for ML frameworks like TensorFlow, JAX, and others.</p>
<p>A cornerstone of XLA is the HLO (High Level Optimizer) IR, which offers a
carefully fixed selected list of operations, mostly orthogonal to each other. It
provides an efficient optimizer for computations expressed with this set of
operations and generate codes for hardware platforms like CPU, GPU, and TPUs.
Its goal is to provide a uniform interface to compile and execute these
optimized HLO programs independently of the targeted device. It is not a
front-end ML system like TensorFlow or JAX, rather it is a backend framework
that optimizes HLO and lowers to machine code.</p>
<p>The HLO set of operations is closed and has well defined semantics. HLO
operations operate on immutable Tensors with static shapes (actually bounded
shapes to be exact) and explicit broadcasts.</p>
<p><a href="https://mlir.llvm.org/">MLIR</a> is a compiler infrastructure which intends to
come with "battery included", as such it intends to provide all the blocks
required to assemble graph optimization and codegen pipelines. The longer term
roadmap for MLIR is to provide a
<a href="https://llvm.discourse.group/c/mlir/MLIR-TCP-WG/36">Tensor Compute Primitive</a>
(TCP) dialect, which should hopefully be general enough to model what HLO
represents today (see
<a href="https://drive.google.com/open?id=1iljcpTQ5NPaMfGpoPDFml1XkYxjK_6A4">slides</a> and
<a href="https://drive.google.com/open?id=1jSPa8TwPKUt0WuLquGc8OgSUVYJHMvWZ">recording</a>
for a technical discussion on this topic).</p>
<p>The work on MLIR-HLO can be seen as a stepping stone towards building TCP, while
integrating intermediate components into XLA itself by relying on the
well-proven HLO IR and introducing more pieces from upstream MLIR
(<a href="https://mlir.llvm.org/docs/Dialects/Linalg/">Linalg</a>,
<a href="https://mlir.llvm.org/docs/Dialects/Vector/">Vector</a>,
<a href="https://mlir.llvm.org/docs/Dialects/GPU/">GPU</a> dialect, ...).
<a href="https://www.tensorflow.org/mlir/xla_gpu_codegen">This document</a> provides more
information on the current migration of the XLA GPU codegen.</p>
<h2>MLIR Dialects for XLA-style compilation</h2>
<p>This repository defines three dialects to support a HLO-like compilation
pipeline using MLIR:</p>
<ul>
<li><code>chlo</code>: the "client" HLO dialect, intended to be closer to the frontend
    (including implicit broadcast semantics).</li>
<li><code>mhlo</code>: "meta"-HLO dialect ; similar to <code>xla_hlo</code>, but with extensions for
    dynamic shape support.</li>
<li><code>lmhlo</code>: "late"-"meta"-HLO, it is the IR after buffer allocation is
    performed. In XLA the buffer allocation is a side-data structure which keeps
    track of these informations, while this separate dialect materializes it in
    the IR.</li>
</ul>
<p>We describe these in more details below.</p>
<h3>HLO Client Dialect: <code>chlo</code>.</h3>
<ul>
<li>It was originally designed to map the
    <a href="https://www.tensorflow.org/xla/operation_semantics">XLA client APIs</a> (e.g.,
    ops supports implicit broadcast and roughly modeled on XlaBuilder API)
    modulo support for dynamic shapes and additional ops required to support
    dynamic client side HLOs.</li>
<li>Ops can be from either the XlaBuilder or XLA helper functions can be
    converted into ops (e.g., given ambiguity in what constitutes these ops,
    there is some freedom to decide), the goal of this dialect is to correspond
    close to client level and enable a thin layer between client use and op
    construction (making it cheap to construct and optimizations on the dialect
    close to optimizations on the client ops).</li>
</ul>
<p>Entry:</p>
<ul>
<li>The vast majority of old "client" interactions are via the XlaBuilder APIs.
    These APIs are used by TF2XLA kernels, JAX, PyTorch bridge and directly. The
    legalization path (described below) can also reuse the XlaBuilder's APIs to
    construct XLA Client HLO ops directly (this uses MlirXlaBuilder which is a
    subclass of XlaBuilder).</li>
<li>The other entry point is during legalization from TensorFlow ops in the TF
    Graph Compiler and other tools (e.g., SavedModel lowering and TFCompile).</li>
</ul>
<p>Exit:</p>
<ul>
<li>MHLO</li>
<li>May be exported to xla::HloInstructionProto by invoking the XlaBuilder APIs
    (with regular XlaBuilder)</li>
</ul>
<p>The <code>chlo</code> dialect started originally as mapping to the XLA client Builder APIs.
It enables it to both be constructed and converted back to existing XLA
interfaces using the XlaBuilder API. Due to the way that translation into and
out of the dialect works, there is no expectation that this dialect roundtrips
to XLA (e.g., it is only intended to be translated to MLIR and then legalized to
another dialect or translated to HloInstructionProto).</p>
<p>The export approach of reusing the XlaBuilders enables reusing a lot of logic
that was already implemented in terms of computing shapes, inserting broadcasts
etc.</p>
<p>An important topic here is that XLA Client HLO ops are not a well defined set.
And in particular what some would consider helper functions, others would
consider ops. It should be easy to move between these and so define a new op
along with the helper function or autogenerate the helper functions from the
descriptions of the ops. For the former, a simple approach would be to simply
consider the context in which the op is being constructed and if an MLIR one,
construct a op in the client dialect instead of further calls into XlaBuilder.
The latter could be implemented by adding the op and a legalization of the op to
other known ops, from which a helper function can get generated that could be
used as regular.</p>
<p>Status: Exists but need to be cleaned up.</p>
<h3>Meta HLO Dialect <code>mhlo</code></h3>
<ul>
<li>Dialect is closer to current HLO server ops (e.g., no implicit broadcast)</li>
<li>MHLO dialect where we can deviate from the requirements of the client or
    server dialect, in particular:<ul>
<li>Control flow ops with implicit capture to enable simpler optimizations
    (e.g., generic LICM, unroll &amp; jam, etc.)</li>
<li>Multiple results ops (e.g., no tuples)</li>
<li>More ops (for example, unique op or assert op), and ops that don't need
    to be added to either client or server dialect.</li>
<li>Op set not constrained by implementation (e.g., hlo.add operating on say
    i79 or !mydialect.weird_type is allowed even though no XLA backend
    supports it). Verification on types happening at the boundaries.</li>
<li>It does not need to preserve some deprecated XLA constructs (e.g.
    stateful RNG HLO).</li>
<li>More dynamic shape support ops without need for updating all
    users/backends.</li>
</ul>
</li>
<li>This dialect enables evolving HLO independently from XLA in order to
    experiment with features we'd like to upstream in MLIR TCP. In particular it
    intends to be user-extensible through
    <a href="https://mlir.llvm.org/docs/Interfaces/">interfaces</a>.</li>
<li>It should have no TensorFlow, or proto, or other Google internal
    dependencies.</li>
<li>It need not be a complete superset of ops compared to XLA HLO dialect.</li>
</ul>
<p>Entry:</p>
<ul>
<li>Legalization from <code>chlo</code> dialect or conversion from XLA HLO.</li>
<li>Directly emitted from TF Graph Compiler;</li>
<li>Builder call (e.g., EDSL);</li>
</ul>
<p>Exit:</p>
<ul>
<li>LMHLO, Linalg IREE, directly used in codegen.</li>
<li>XLA HLO.</li>
</ul>
<p>The MHLO dialect has no direct export format, it is only meant as an
intermediate optimization dialect/format. It is also where we can experiment
cheaply with new ops. This format will be where the representation would differ
from existing endpoints.</p>
<p>Status: Exists but need to be cleaned up and evolved, in particular with respect
to supporting dynamic shapes.</p>
<p>MHLO differs from XLA HLO op set in multiple ways, including:
1. MHLO While accepts multiple operands and may produce multiple results
   instead;</p>
<h3>LMHLO</h3>
<p>LMHLO corresponds to late <code>mhlo</code> and operates on buffer domain (e.g., memref)
with side-effecting operations. The lowering from <code>mhlo</code> dialect proceeds by way
of scheduling, memory and buffer allocation. The current mapping is directly on
XLA Client HLOs but without implicit broadcast and with operation on memrefs.
This dialect will instead be rebased on <code>mhlo</code> dialect but operating on buffers
still.</p>
<p>Entry:</p>
<ul>
<li>Post buffer assignment on <code>mhlo</code> dialect, or from XLA after buffer
    assignment.</li>
</ul>
<p>Exit:</p>
<ul>
<li>Codegen (LLVM IR in the common cases at the moment)</li>
</ul>
<h2>End-to-End pipeline</h2>
<p>TODO</p>
<h2>Alternative build setups</h2>
<h3>Building Python API</h3>
<p>Building the MHLO Python API requires building as an LLVM external project.
The below instructions presume that you have this <code>mlir-hlo</code> repo and an
<code>llvm-project</code> repo checked out side by side.</p>
<p>Note that the python package produced by this procedure includes the <code>mlir</code>
package and is not suitable for deployment as-is (but it can be included into
a larger aggregate).</p>
<p>```
mkdir build &amp;&amp; cd build
cmake -GNinja -B. ${LLVM_SRC_DIR}/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=mlir_hlo \
    -DLLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR=${MLIR_HLO_SRC_DIR} \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DPython3_EXECUTABLE=$(which python) \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMHLO_ENABLE_BINDINGS_PYTHON=ON</p>
<p>ninja MLIRHLOPythonModules
export PYTHONPATH=$PWD/tools/mlir_hlo/python_packages/mlir_hlo
python -c "import mlir.dialects.mhlo"
```</p>
<h2>External projects that depend on mlir-hlo</h2>
<p>External projects that need to depend on <code>mlir-hlo</code> (for example via a git
submodule) can use the following setting in their cmake configuration in order
for <code>find_package(MHLO)</code> to import all mlir-hlo cmake targets into their build
setup and have access to the required include and lib variables (see generated
<code>MHLOConfig.cmake</code>).</p>
<p><code>...
   -DMHLO_DIR=&lt;path to mlir-hlo build dir&gt;/lib/cmake/mlir-hlo
   ...</code></p>