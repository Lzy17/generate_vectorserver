<h1>Torchdynamo Benchmarks</h1>
<h2>What We Benchmark</h2>
<p>TorchDynamo provides a benchmark harness that takes care of uniformly benchmarking different models.  It interleaves runs of eager and dynamo to avoid machine noise/variability issues, and reports results based on medians along with P-values.</p>
<p>The runner integrates with models from TorchBenchmark, HuggingFace and TIMM suites and covers both training and inference.</p>
<p>The infrastructure allows us to specify a loss function. For torchbench models, we use .sum().backward() call in place of the native loss function. For TIMM models, we use a CrossEntropy loss. And HF models contain a loss function inside the model itself, so we don't need any special loss computation handling.</p>
<p>Training benchmarks approximate training by running the model forward, computing loss, running backward, and then the optimizer (SGD). Note: the optimizer is currently not compiled by Torchdynamo.</p>
<p>Inference benchmarks and Training benchmarks measure correctness by comparing dynamo and eager model outputs given fixed inputs and seeds.</p>
<h2>Setup</h2>
<h3>Machine</h3>
<p>We run benchmarks on AWS machines (p4d.24xlarge) using 8xNVidia A100 40GB cards.  We suggest using Cuda 11.6 for consistency.</p>
<h3>Benchmarks</h3>
<p>Make sure to carefully follow the <a href="https://github.com/pytorch/benchmark#installation">torchbench installation</a> instructions, taking care to build the auxiliary libraries (torchvision, torchtext) from a matching version to your pytorch version.</p>
<p>For HF and TIMM models, the scripts already install the transformers and timm package respectively on the first run.</p>
<h2>Runbook</h2>
<h3>Basic Usage</h3>
<p>There are a lot of flags in the benchmark runner, and it can be confusing to know which settings to use or what machine to run it on.  In order to support apples-to-apples comparison, we have provided the following 'standard' settings in <code>runner.py</code>. This script is a wrapper over the common benchmarking infrastructure and simplifies the flags. We will continually update <code>runner.py</code> with the latest and most relevant compilers for training and inference. It also provides some graph utilities to visualize and compare results. Some of the example commands are</p>
<p><strong>Inference Commands</strong>
* Inference compilers on torchbench models - <code>python benchmarks/dynamo/runner.py --suites=torchbench --inference --dtypes=float16</code>
* Inductor Inference compiler on torchbench models - <code>python benchmarks/dynamo/runner.py --suites=torchbench --inference --dtypes=float16 --compilers=inductor</code></p>
<p><strong>Training Commands</strong>
* Training compilers on TIMM models - <code>python benchmarks/dynamo/runner.py --suites=timm_models --training --dtypes=float32 --output-dir=timm_logs</code>
* AOTAutograd Training compiler on TIMM models - <code>python benchmarks/dynamo/runner.py --suites=timm_models --training --dtypes=float32 --compilers=aot_nvfuser --output-dir=timm_logs</code>
* Inductor Training compiler on TIMM models - <code>python benchmarks/dynamo/runner.py --suites=timm_models --training --dtypes=float32 --compilers=inductor --output-dir=timm_logs</code></p>
<p>Running runner.py generates a file named <code>run.sh</code>. This file contains the actual commands that invoke the common benchmarking infrastructure with the appropriate flags. Which brings us to the advanced usage.</p>
<h3>Advanced Usage</h3>
<p>One could directly call <code>torchbench.py</code>, <code>huggingface.py</code> or <code>timm_models.py</code> with the necessary flags. There are a lot of flags in the benchmarks runner. Some of the examples are as follows. These are subject to change.</p>
<p><strong>Inference Commands</strong>
* TorchScript (with TorchDynamo capture) NVFuser Inference - <code>python benchmarks/dynamo/torchbench.py -dcuda -n100 --speedup-dynamo-ts --performance</code>
* TorchInductor CUDA Graphs Inference - <code>python benchmarks/dynamo/torchbench.py -dcuda --float32 -n50 --inductor --performance</code></p>
<p><strong>Training Commands</strong>
* Torchscript (with TorchDynamo capture) NVFuser Training - <code>python benchmarks/dynamo/torchbench.py --float32 -dcuda --training --nvfuser --speedup-dynamo-ts --performance</code>
* TorchInductor CUDA Graphs Training - <code>python benchmarks/dynamo/torchbench.py --float32 -dcuda --training --inductor --performance</code></p>
<p>Above commands are for torchbench models. You can simply replace <code>torchbench.py</code> with <code>huggingface.py</code> for HF models, and <code>timm_model.py</code> for TIMM models.</p>