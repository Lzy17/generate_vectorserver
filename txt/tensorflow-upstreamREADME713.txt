<p align="center">
  <img width="200" src="./g3doc/images/xlalogo.png"/>
</p>

<p>XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear
algebra that optimizes TensorFlow computations. See the
<a href="./g3doc/index.md">documentation</a>.</p>
<p>This directory is currently migrating to <a href="https://github.com/openxla/">OpenXLA</a>
and will be the root of the <a href="https://github.com/openxla/xla">openxla/xla</a>
repository.</p>
<p>== Directory Structure ==</p>
<p>We're currently re-organizing the directory structure, the end result should be
that no sources are directly present at the top-level. Here is the current plan
for the directory layout:</p>
<ul>
<li>backends/ (created from directories under xla/service)<ul>
<li>cpu/</li>
<li>gpu/</li>
<li>interpreter/</li>
<li>...</li>
</ul>
</li>
<li>hlo/ (created from xla/service/ mostly, no sources expected directly here)<ul>
<li>client/ (created from xla/client)</li>
<li>evaluator/ (created from the relevant files in xla/service)</li>
<li>experimental/ (created from xla/experimental)</li>
<li>ir/ (created from the relevant files in xla/service)</li>
<li>python/ (created from xla/python)</li>
<li>tests/ (created from xla/tests)</li>
<li>transforms/ (created from the relevant files in xla/service)</li>
<li>utils/ (created from the relevant files in xla/service)</li>
</ul>
</li>
<li>mlir/ (also exported as the root of https://github.com/tensorflow/mlir-hlo
    and building with CMake)<ul>
<li>CMakeLists.txt (just like now for mlir-hlo repo).</li>
<li>backends/ (same as xla/backends/ but for the MLIR specific bits: this is
    a short-term solution pending more convergence / XLA Next)<ul>
<li>cpu</li>
<li>gpu (populated from /compiler/xla/mlir/transforms/gpu/passes.td,
    will contain all the glue for e2e GPU compilation)</li>
</ul>
</li>
<li>bindings/<ul>
<li>c/ (bootstrapped from mlir/hlo/{include,lib}/mlir-hlo-c)</li>
<li>python/ (bootstrapped from mlir/hlo/python, should talk about some
    low-level LAX?)</li>
</ul>
</li>
<li>integration_tests/ (to be defined / refined)</li>
<li>tools/ (fuzzer, ir-reducer, interpreter/evaluator)</li>
<li>transforms/ (generic / cross dialect transforms)</li>
<li>utils/</li>
</ul>
</li>
<li>// below are dialects and transforms folders<ul>
<li>framework/ (moved from compiler/mlir/xla/ir/xla_framework_ops.td)</li>
<li>gml_st<ul>
<li>gmlst-opt.cc</li>
<li>gmlst-runner.cc (runner tool that can execute IR at ~gmlst level)</li>
<li>ir/</li>
<li>integration_test (tests that run things: Tensor(s) in -&gt; Tensor(s)
    out)</li>
<li>test (IR -&gt; IR tests for passes interaction)</li>
<li>transforms/<ul>
<li>bufferize_tiled_loop/<ul>
<li>bufferize_tiled_loop.cc</li>
<li>bufferize_tiled_loop.h</li>
</ul>
</li>
<li>...</li>
</ul>
</li>
</ul>
</li>
<li>lhlo_gpu/</li>
<li>mhlo/<ul>
<li>mhlo-opt.cc</li>
<li>analysis/<ul>
<li>dataflow/<ul>
<li>dataflow.h</li>
<li>dataflow.cc</li>
<li>test_pass.cc // test_only target, linked into opt tool for
    testing only.</li>
</ul>
</li>
</ul>
</li>
<li>integration_test (tests that run things: Tensor(s) in -&gt; Tensor(s)
    out)</li>
<li>ir/ (dialect definition)</li>
<li>test (IR -&gt; IR tests for passes interaction)</li>
<li>transforms/<ul>
<li>materialize_broadcasts/<ul>
<li>materialize_broadcasts.cc</li>
<li>materialize_broadcasts.h // headers stays with the source</li>
<li>broadcast_analysis.{cc, h} // private analysis/utils needed
    for this pass</li>
<li>test/ (.mlir unit-tests are collocated with the pass
    itself).</li>
</ul>
</li>
<li>…</li>
<li>passes.td // enables group registration for all passes.</li>
</ul>
</li>
<li>utils/</li>
</ul>
</li>
<li>thlo/</li>
<li>runtime/</li>
</ul>
</li>
<li>pjrt/ (created from xla/pjrt)</li>
<li>rpc/ (created from xla/rpc)</li>
<li>runtime/</li>
<li>stream_executor/ (moved from TensorFlow)</li>
<li>third_party/ (vendoring of TSL base library)</li>
<li>tools/ (created from mlir/hlo/tools and xla/tools)</li>
<li>translate/ (StableHLO to MHLO, MHLO to HLO, HLO to MHLO, MHLO to TOSA)</li>
</ul>