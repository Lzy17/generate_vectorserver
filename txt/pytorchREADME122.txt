<h1>Fast RNN benchmarks</h1>
<p>Benchmarks for TorchScript models</p>
<p>For most stable results, do the following:
- Set CPU Governor to performance mode (as opposed to energy save)
- Turn off turbo for all CPUs (assuming Intel CPUs)
- Shield cpus via <code>cset shield</code> when running benchmarks.</p>
<p>Some of these scripts accept command line args but most of them do not because
I was lazy. They will probably be added sometime in the future, but the default
sizes are pretty reasonable.</p>
<h2>Test fastrnns (fwd + bwd) correctness</h2>
<p>Test the fastrnns benchmarking scripts with the following:
<code>python -m fastrnns.test</code>
or run the test independently:
<code>python -m fastrnns.test --rnns jit</code></p>
<h2>Run benchmarks</h2>
<p><code>python -m fastrnns.bench</code></p>
<p>should give a good comparison, or you can specify the type of model to run</p>
<p><code>python -m fastrnns.bench --rnns cudnn aten jit --group rnns</code></p>
<h2>Run model profiling, calls nvprof</h2>
<p><code>python -m fastrnns.profile</code></p>
<p>should generate nvprof file for all models somewhere.
you can also specify the models to generate nvprof files separately:</p>
<p><code>python -m fastrnns.profile --rnns aten jit</code></p>
<h3>Caveats</h3>
<p>Use Linux for the most accurate timing. A lot of these tests only run
on CUDA.</p>