<h1>PyTorch Benchmarks</h1>
<p>This folder contains scripts that produce reproducible timings of various PyTorch features.</p>
<p>It also provides mechanisms to compare PyTorch with other frameworks.</p>
<h2>Setup environment</h2>
<p>Make sure you're on a machine with CUDA, torchvision, and pytorch installed. Install in the following order:
```</p>
<h1>Install torchvision. It comes with the pytorch stable release binary</h1>
<p>conda install pytorch torchvision -c pytorch</p>
<h1>Install the latest pytorch master from source.</h1>
<h1>It should supersede the installation from the release binary.</h1>
<p>cd $PYTORCH_HOME
python setup.py build develop</p>
<h1>Check the pytorch installation version</h1>
<p>python -c "import torch; print(torch.<strong>version</strong>)"
```</p>
<h2>Benchmark List</h2>
<p>Please refer to each subfolder to discover each benchmark suite</p>
<ul>
<li><a href="fastrnns/README.md">Fast RNNs benchmarks</a></li>
</ul>