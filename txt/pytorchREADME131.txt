<h1><code>__torch_function__</code> micro-benchmarks</h1>
<p>This benchmark suite provides a systemic way to measure the performance of <code>__torch_function__</code> overhead.</p>
<h2>Getting started</h2>
<h3>Initial Setup</h3>
<p>Install <code>py-spy</code> by doing:</p>
<p><code>bash
pip install py-spy</code></p>
<p>Note that more extensive documentation on using <code>py-spy</code> is available in <code>CONTRIBUTING.md</code>.</p>
<h3>Running the benchmark</h3>
<p>Run one of the following commands in the terminal, with the working directory being <code>${PYTORCH_CLONE_DIR}/benchmarks/overrides_benchmark</code>:</p>
<p>```bash</p>
<h1>Benchmark all the cases</h1>
<p>python bench.py</p>
<h1>Flame graph pertaining to each case.</h1>
<p>py-spy record -o tensor.svg --native -- python pyspybench.py Tensor
py-spy record -o subtensor.svg --native -- python pyspybench.py SubTensor
py-spy record -o overridden.svg --native -- python pyspybench.py WithTorchFunction
py-spy record -o suboverridden.svg --native -- python pyspybench.py SubWithTorchFunction
```</p>
<p>Here is a brief overview of what the results should look like, if run correctly:</p>
<ul>
<li>Overhead for <code>torch</code> functions when run on <code>torch.Tensor</code> objects is on the order of 2 μs.</li>
<li><code>__torch_function__</code> should add zero overhead for <code>torch.Tensor</code> inputs, a small overhead for subclasses of <code>torch.Tensor</code>, and a couple of microseconds for <code>Tensor</code>-likes with <code>__torch_function__</code>.</li>
<li>Changing the dispatching mechanism may result in changes that are on the order of 100 ns, which are hard to detect due to noise, but important.</li>
</ul>
<h2>Reporting benchmark results</h2>
<p>When modifying any of the machinery around <code>__torch_function__</code>, run the benchmark for both the feature branch and the point it diverges from <code>master</code>. For each of these:</p>
<ul>
<li>Run <code>bench.py</code>, and include the output in your result.</li>
<li>For each case where <code>bench.py</code> shows a regression, run the commands described above, prefixing the output SVG filename (the input to the <code>-o</code> switch) with <code>base-</code> or <code>branch-</code> depending on the commit you are running the benchmark on.</li>
<li>For each SVG, open it in the browser, take a screenshot and include it in your result. Also include a ZIP file with all SVGs thus produced included.</li>
</ul>