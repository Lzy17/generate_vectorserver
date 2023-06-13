<h1>Benchmarking tool for the autograd API</h1>
<p>This folder contain a set of self-contained scripts that allow to benchmark the autograd with different common models.
It is designed to run the benchmark before and after your change and will generate a table to share on the PR.</p>
<p>To do so, you can use <code>functional_autograd_benchmark.py</code> to run the benchmarks before your change (using as output <code>before.txt</code>) and after your change (using as output <code>after.txt</code>).
You can then use <code>compare.py</code> to get a markdown table comparing the two runs.</p>
<p>The default arguments of <code>functional_autograd_benchmark.py</code> should be used in general. You can change them though to force a given device or force running even the (very) slow settings.</p>
<h3>Sample usage</h3>
<p>```bash</p>
<h1>Make sure you compile pytorch in release mode and with the same flags before/after</h1>
<p>export DEBUG=0</p>
<h1>When running on CPU, it might be required to limit the number of cores to avoid oversubscription</h1>
<p>export OMP_NUM_THREADS=10</p>
<h1>Compile pytorch with the base revision</h1>
<p>git checkout master
python setup.py develop</p>
<h1>Install dependencies:</h1>
<h1>Scipy is required by detr</h1>
<p>pip install scipy</p>
<h1>Run the benchmark for the base</h1>
<h1>This will use the GPU if available.</h1>
<p>pushd benchmarks/functional_autograd_benchmark
python functional_autograd_benchmark.py --output before.txt</p>
<h1>Compile pytorch with your change</h1>
<p>popd
git checkout your_feature_branch
python setup.py develop</p>
<h1>Run the benchmark for the new version</h1>
<p>pushd benchmarks/functional_autograd_benchmark
python functional_autograd_benchmark.py --output after.txt</p>
<h1>Get the markdown table that you can paste in your github PR</h1>
<p>python compare.py</p>
<p>popd</p>
<p>```</p>
<h3>Files in this folder:</h3>
<ul>
<li><code>functional_autograd_benchmark.py</code> is the main entry point to run the benchmark.</li>
<li><code>compare.py</code> is the entry point to run the comparison script that generates a markdown table.</li>
<li><code>torchaudio_models.py</code> and <code>torchvision_models.py</code>  contains code extracted from torchaudio and torchvision to be able to run the models without having a specific version of these libraries installed.</li>
<li><code>ppl_models.py</code>, <code>vision_models.py</code> and <code>audio_text_models.py</code> contain all the getter functions used for the benchmark.</li>
</ul>
<h3>Benchmarking against <code>functorch</code></h3>
<p>```bash</p>
<h1>Install stable functorch:</h1>
<p>pip install functorch</p>
<h1>or install from source:</h1>
<p>pip install git+https://github.com/pytorch/functorch</p>
<h1>Run the benchmark for the base</h1>
<h1>This will use the GPU if available.</h1>
<p>pushd benchmarks/functional_autograd_benchmark
python functional_autograd_benchmark.py --output bench-with-functorch.txt
```</p>