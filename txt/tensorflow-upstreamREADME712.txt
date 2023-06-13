<h1>Performance benchmarks for MLIR based code generation</h1>
<p>These benchmarks compare performance of Tensorflow -&gt; LLVM code generation
with Eigen. These benchmarks are based on the Google Benchmark library and
can be integrated with performance monitoring tools.</p>
<h2>Running benchmarks</h2>
<p><code>bazel run -c opt --cpu=haswell \
  :cwise_op_tanh_benchmark -- --benchmark_filter="f32/10k"</code></p>
<h2>Using perf and pprof with these benchmarks</h2>
<ol>
<li>
<p>Record perf profile
<code>perf record -k 1 -o /tmp/perf.data --        \
  bazel run -c opt --cpu=haswell -copt=-gmlt \
  :cwise_op_tanh_benchmark -- --benchmark_filter="f32/10k"</code></p>
</li>
<li>
<p>Inject data from the JIT compiled functions
<code>perf inject -j -v -i /tmp/perf.data -o /tmp/perf.data.jit</code></p>
</li>
<li>
<p>Report perf data</p>
</li>
</ol>
<p><code>perf report -i /tmp/perf.data.jit</code></p>
<p>or</p>
<p><code>pprof -flame -nodecount=10000 /tmp/perf.data.jit</code></p>
<!-- BEGIN GOOGLE-INTERNAL -->
<h2>Running benchmarks using perflab and benchy</h2>
<ol>
<li>go/benchy</li>
<li>go/perflab</li>
</ol>
<p><code>benchy                                                                        \
  --reference=${reference} --cpu=haswell --runs=20 --benchmark_filter=all     \
  --perflab --borg_constraints="platform_family_genus_cpu=indus-skylake-2000" \
  third_party/tensorflow/compiler/mlir/tfrt/benchmarks:cwise_op_tanh_benchmark</code></p>
<p>As of Q1 2021 <code>indus-skylake-2000</code> is the machine of the day, and roughly 60% of
the fleet cycles are executed on Skylakes.</p>
<p>Reference can be: 1. Cl number to test agains another pending change 2. <code>srcfs</code>
to test agains the g3 head 3. Another client number to test local changes
without exporting them <!-- END GOOGLE-INTERNAL --></p>