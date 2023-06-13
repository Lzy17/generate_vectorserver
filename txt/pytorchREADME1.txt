<p><img alt="PyTorch Logo" src="https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.png" /></p>
<hr />
<p>PyTorch is a Python package that provides two high-level features:
- Tensor computation (like NumPy) with strong GPU acceleration
- Deep neural networks built on a tape-based autograd system</p>
<p>You can reuse your favorite Python packages such as NumPy, SciPy, and Cython to extend PyTorch when needed.</p>
<p>Our trunk health (Continuous Integration signals) can be found at <a href="https://hud.pytorch.org/ci/pytorch/pytorch/main">hud.pytorch.org</a>.</p>
<!-- toc -->

<ul>
<li><a href="#more-about-pytorch">More About PyTorch</a></li>
<li><a href="#a-gpu-ready-tensor-library">A GPU-Ready Tensor Library</a></li>
<li><a href="#dynamic-neural-networks-tape-based-autograd">Dynamic Neural Networks: Tape-Based Autograd</a></li>
<li><a href="#python-first">Python First</a></li>
<li><a href="#imperative-experiences">Imperative Experiences</a></li>
<li><a href="#fast-and-lean">Fast and Lean</a></li>
<li><a href="#extensions-without-pain">Extensions Without Pain</a></li>
<li><a href="#installation">Installation</a></li>
<li><a href="#binaries">Binaries</a><ul>
<li><a href="#nvidia-jetson-platforms">NVIDIA Jetson Platforms</a></li>
</ul>
</li>
<li><a href="#from-source">From Source</a><ul>
<li><a href="#prerequisites">Prerequisites</a></li>
<li><a href="#install-dependencies">Install Dependencies</a></li>
<li><a href="#get-the-pytorch-source">Get the PyTorch Source</a></li>
<li><a href="#install-pytorch">Install PyTorch</a></li>
<li><a href="#adjust-build-options-optional">Adjust Build Options (Optional)</a></li>
</ul>
</li>
<li><a href="#docker-image">Docker Image</a><ul>
<li><a href="#using-pre-built-images">Using pre-built images</a></li>
<li><a href="#building-the-image-yourself">Building the image yourself</a></li>
</ul>
</li>
<li><a href="#building-the-documentation">Building the Documentation</a></li>
<li><a href="#previous-versions">Previous Versions</a></li>
<li><a href="#getting-started">Getting Started</a></li>
<li><a href="#resources">Resources</a></li>
<li><a href="#communication">Communication</a></li>
<li><a href="#releases-and-contributing">Releases and Contributing</a></li>
<li><a href="#the-team">The Team</a></li>
<li><a href="#license">License</a></li>
</ul>
<!-- tocstop -->

<h2>More About PyTorch</h2>
<p>At a granular level, PyTorch is a library that consists of the following components:</p>
<p>| Component | Description |
| ---- | --- |
| <a href="https://pytorch.org/docs/stable/torch.html"><strong>torch</strong></a> | A Tensor library like NumPy, with strong GPU support |
| <a href="https://pytorch.org/docs/stable/autograd.html"><strong>torch.autograd</strong></a> | A tape-based automatic differentiation library that supports all differentiable Tensor operations in torch |
| <a href="https://pytorch.org/docs/stable/jit.html"><strong>torch.jit</strong></a> | A compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code  |
| <a href="https://pytorch.org/docs/stable/nn.html"><strong>torch.nn</strong></a> | A neural networks library deeply integrated with autograd designed for maximum flexibility |
| <a href="https://pytorch.org/docs/stable/multiprocessing.html"><strong>torch.multiprocessing</strong></a> | Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training |
| <a href="https://pytorch.org/docs/stable/data.html"><strong>torch.utils</strong></a> | DataLoader and other utility functions for convenience |</p>
<p>Usually, PyTorch is used either as:</p>
<ul>
<li>A replacement for NumPy to use the power of GPUs.</li>
<li>A deep learning research platform that provides maximum flexibility and speed.</li>
</ul>
<p>Elaborating Further:</p>
<h3>A GPU-Ready Tensor Library</h3>
<p>If you use NumPy, then you have used Tensors (a.k.a. ndarray).</p>
<p><img alt="Tensor illustration" src="./docs/source/_static/img/tensor_illustration.png" /></p>
<p>PyTorch provides Tensors that can live either on the CPU or the GPU and accelerates the
computation by a huge amount.</p>
<p>We provide a wide variety of tensor routines to accelerate and fit your scientific computation needs
such as slicing, indexing, mathematical operations, linear algebra, reductions.
And they are fast!</p>
<h3>Dynamic Neural Networks: Tape-Based Autograd</h3>
<p>PyTorch has a unique way of building neural networks: using and replaying a tape recorder.</p>
<p>Most frameworks such as TensorFlow, Theano, Caffe, and CNTK have a static view of the world.
One has to build a neural network and reuse the same structure again and again.
Changing the way the network behaves means that one has to start from scratch.</p>
<p>With PyTorch, we use a technique called reverse-mode auto-differentiation, which allows you to
change the way your network behaves arbitrarily with zero lag or overhead. Our inspiration comes
from several research papers on this topic, as well as current and past work such as
<a href="https://github.com/twitter/torch-autograd">torch-autograd</a>,
<a href="https://github.com/HIPS/autograd">autograd</a>,
<a href="https://chainer.org">Chainer</a>, etc.</p>
<p>While this technique is not unique to PyTorch, it's one of the fastest implementations of it to date.
You get the best of speed and flexibility for your crazy research.</p>
<p><img alt="Dynamic graph" src="https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/dynamic_graph.gif" /></p>
<h3>Python First</h3>
<p>PyTorch is not a Python binding into a monolithic C++ framework.
It is built to be deeply integrated into Python.
You can use it naturally like you would use <a href="https://www.numpy.org/">NumPy</a> / <a href="https://www.scipy.org/">SciPy</a> / <a href="https://scikit-learn.org">scikit-learn</a> etc.
You can write your new neural network layers in Python itself, using your favorite libraries
and use packages such as <a href="https://cython.org/">Cython</a> and <a href="http://numba.pydata.org/">Numba</a>.
Our goal is to not reinvent the wheel where appropriate.</p>
<h3>Imperative Experiences</h3>
<p>PyTorch is designed to be intuitive, linear in thought, and easy to use.
When you execute a line of code, it gets executed. There isn't an asynchronous view of the world.
When you drop into a debugger or receive error messages and stack traces, understanding them is straightforward.
The stack trace points to exactly where your code was defined.
We hope you never spend hours debugging your code because of bad stack traces or asynchronous and opaque execution engines.</p>
<h3>Fast and Lean</h3>
<p>PyTorch has minimal framework overhead. We integrate acceleration libraries
such as <a href="https://software.intel.com/mkl">Intel MKL</a> and NVIDIA (<a href="https://developer.nvidia.com/cudnn">cuDNN</a>, <a href="https://developer.nvidia.com/nccl">NCCL</a>) to maximize speed.
At the core, its CPU and GPU Tensor and neural network backends
are mature and have been tested for years.</p>
<p>Hence, PyTorch is quite fast — whether you run small or large neural networks.</p>
<p>The memory usage in PyTorch is extremely efficient compared to Torch or some of the alternatives.
We've written custom memory allocators for the GPU to make sure that
your deep learning models are maximally memory efficient.
This enables you to train bigger deep learning models than before.</p>
<h3>Extensions Without Pain</h3>
<p>Writing new neural network modules, or interfacing with PyTorch's Tensor API was designed to be straightforward
and with minimal abstractions.</p>
<p>You can write new neural network layers in Python using the torch API
<a href="https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html">or your favorite NumPy-based libraries such as SciPy</a>.</p>
<p>If you want to write your layers in C/C++, we provide a convenient extension API that is efficient and with minimal boilerplate.
No wrapper code needs to be written. You can see <a href="https://pytorch.org/tutorials/advanced/cpp_extension.html">a tutorial here</a> and <a href="https://github.com/pytorch/extension-cpp">an example here</a>.</p>
<h2>Installation</h2>
<h3>Binaries</h3>
<p>Commands to install binaries via Conda or pip wheels are on our website: <a href="https://pytorch.org/get-started/locally/">https://pytorch.org/get-started/locally/</a></p>
<h4>NVIDIA Jetson Platforms</h4>
<p>Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are provided <a href="https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048">here</a> and the L4T container is published <a href="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch">here</a></p>
<p>They require JetPack 4.2 and above, and <a href="https://github.com/dusty-nv">@dusty-nv</a> and <a href="https://github.com/ptrblck">@ptrblck</a> are maintaining them.</p>
<h3>From Source</h3>
<h4>Prerequisites</h4>
<p>If you are installing from source, you will need:
- Python 3.8 or later (for Linux, Python 3.8.1+ is needed)
- A C++17 compatible compiler, such as clang</p>
<p>We highly recommend installing an <a href="https://www.anaconda.com/distribution/#download-section">Anaconda</a> environment. You will get a high-quality BLAS library (MKL) and you get controlled dependency versions regardless of your Linux distro.</p>
<p>If you want to compile with CUDA support, install the following (note that CUDA is not supported on macOS)
- <a href="https://developer.nvidia.com/cuda-downloads">NVIDIA CUDA</a> 11.0 or above
- <a href="https://developer.nvidia.com/cudnn">NVIDIA cuDNN</a> v7 or above
- <a href="https://gist.github.com/ax3l/9489132">Compiler</a> compatible with CUDA</p>
<p>Note: You could refer to the <a href="https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf">cuDNN Support Matrix</a> for cuDNN versions with the various supported CUDA, CUDA driver and NVIDIA hardware</p>
<p>If you want to disable CUDA support, export the environment variable <code>USE_CUDA=0</code>.
Other potentially useful environment variables may be found in <code>setup.py</code>.</p>
<p>If you are building for NVIDIA's Jetson platforms (Jetson Nano, TX1, TX2, AGX Xavier), Instructions to install PyTorch for Jetson Nano are <a href="https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/">available here</a></p>
<p>If you want to compile with ROCm support, install
- <a href="https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html">AMD ROCm</a> 4.0 and above installation
- ROCm is currently supported only for Linux systems.</p>
<p>If you want to disable ROCm support, export the environment variable <code>USE_ROCM=0</code>.
Other potentially useful environment variables may be found in <code>setup.py</code>.</p>
<h4>Install Dependencies</h4>
<p><strong>Common</strong></p>
<p>```bash
conda install cmake ninja</p>
<h1>Run this command from the PyTorch directory after cloning the source code using the “Get the PyTorch Source“ section below</h1>
<p>pip install -r requirements.txt
```</p>
<p><strong>On Linux</strong></p>
<p>```bash
conda install mkl mkl-include</p>
<h1>CUDA only: Add LAPACK support for the GPU if needed</h1>
<p>conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo</p>
<h1>(optional) If using torch.compile with inductor/triton, install the matching version of triton</h1>
<h1>Run from the pytorch directory after cloning</h1>
<p>make triton
```</p>
<p><strong>On MacOS</strong></p>
<p>```bash</p>
<h1>Add this package on intel x86 processor machines only</h1>
<p>conda install mkl mkl-include</p>
<h1>Add these packages if torch.distributed is needed</h1>
<p>conda install pkg-config libuv
```</p>
<p><strong>On Windows</strong></p>
<p>```bash
conda install mkl mkl-include</p>
<h1>Add these packages if torch.distributed is needed.</h1>
<h1>Distributed package support on Windows is a prototype feature and is subject to changes.</h1>
<p>conda install -c conda-forge libuv=1.39
```</p>
<h4>Get the PyTorch Source</h4>
<p>```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch</p>
<h1>if you are updating an existing checkout</h1>
<p>git submodule sync
git submodule update --init --recursive
```</p>
<h4>Install PyTorch</h4>
<p><strong>On Linux</strong></p>
<p>If you're compiling for AMD ROCm then first run this command:
```bash</p>
<h1>Only run this if you're compiling for ROCm</h1>
<p>python tools/amd_build/build_amd.py
```</p>
<p>Install PyTorch
<code>bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop</code></p>
<blockquote>
<p><em>Aside:</em> If you are using <a href="https://www.anaconda.com/distribution/#download-section">Anaconda</a>, you may experience an error caused by the linker:</p>
<p><code>plaintext
build/temp.linux-x86_64-3.7/torch/csrc/stub.o: file not recognized: file format not recognized
collect2: error: ld returned 1 exit status
error: command 'g++' failed with exit status 1</code></p>
<p>This is caused by <code>ld</code> from the Conda environment shadowing the system <code>ld</code>. You should use a newer version of Python that fixes this issue. The recommended Python version is 3.8.1+.</p>
</blockquote>
<p><strong>On macOS</strong></p>
<p><code>bash
python3 setup.py develop</code></p>
<p><strong>On Windows</strong></p>
<p>Choose Correct Visual Studio Version.</p>
<p>PyTorch CI uses Visual C++ BuildTools, which come with Visual Studio Enterprise,
Professional, or Community Editions. You can also install the build tools from
https://visualstudio.microsoft.com/visual-cpp-build-tools/. The build tools <em>do not</em>
come with Visual Studio Code by default.</p>
<p>If you want to build legacy python code, please refer to <a href="https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda">Building on legacy code and CUDA</a></p>
<p><strong>CPU-only builds</strong></p>
<p>In this mode PyTorch computations will run on your CPU, not your GPU</p>
<p><code>cmd
conda activate
python setup.py develop</code></p>
<p>Note on OpenMP: The desired OpenMP implementation is Intel OpenMP (iomp). In order to link against iomp, you'll need to manually download the library and set up the building environment by tweaking <code>CMAKE_INCLUDE_PATH</code> and <code>LIB</code>. The instruction <a href="https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source">here</a> is an example for setting up both MKL and Intel OpenMP. Without these configurations for CMake, Microsoft Visual C OpenMP runtime (vcomp) will be used.</p>
<p><strong>CUDA based build</strong></p>
<p>In this mode PyTorch computations will leverage your GPU via CUDA for faster number crunching</p>
<p><a href="https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm">NVTX</a> is needed to build Pytorch with CUDA.
NVTX is a part of CUDA distributive, where it is called "Nsight Compute". To install it onto an already installed CUDA run CUDA installation once again and check the corresponding checkbox.
Make sure that CUDA with Nsight Compute is installed after Visual Studio.</p>
<p>Currently, VS 2017 / 2019, and Ninja are supported as the generator of CMake. If <code>ninja.exe</code> is detected in <code>PATH</code>, then Ninja will be used as the default generator, otherwise, it will use VS 2017 / 2019.
<br/> If Ninja is selected as the generator, the latest MSVC will get selected as the underlying toolchain.</p>
<p>Additional libraries such as
<a href="https://developer.nvidia.com/magma">Magma</a>, <a href="https://github.com/oneapi-src/oneDNN">oneDNN, a.k.a. MKLDNN or DNNL</a>, and <a href="https://github.com/mozilla/sccache">Sccache</a> are often needed. Please refer to the <a href="https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers">installation-helper</a> to install them.</p>
<p>You can refer to the <a href="https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat">build_pytorch.bat</a> script for some other environment variables configurations</p>
<p>```cmd
cmd</p>
<p>:: Set the environment variables after you have downloaded and unzipped the mkl package,
:: else CMake would throw an error as <code>Could NOT find OpenMP</code>.
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%</p>
<p>:: Read the content in the previous section carefully before you proceed.
:: [Optional] If you want to override the underlying toolset used by Ninja and Visual Studio with CUDA, please run the following script block.
:: "Visual Studio 2019 Developer Command Prompt" will be run automatically.
:: Make sure you have CMake &gt;= 3.12 before you do this when you use the Visual Studio generator.
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (<code>"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath</code>) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%</p>
<p>:: [Optional] If you want to override the CUDA host compiler
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe</p>
<p>python setup.py develop</p>
<p>```</p>
<h5>Adjust Build Options (Optional)</h5>
<p>You can adjust the configuration of cmake variables optionally (without building first), by doing
the following. For example, adjusting the pre-detected directories for CuDNN or BLAS can be done
with such a step.</p>
<p>On Linux
<code>bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py build --cmake-only
ccmake build  # or cmake-gui build</code></p>
<p>On macOS
<code>bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build --cmake-only
ccmake build  # or cmake-gui build</code></p>
<h3>Docker Image</h3>
<h4>Using pre-built images</h4>
<p>You can also pull a pre-built docker image from Docker Hub and run with docker v19.03+</p>
<p><code>bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest</code></p>
<p>Please note that PyTorch uses shared memory to share data between processes, so if torch multiprocessing is used (e.g.
for multithreaded data loaders) the default shared memory segment size that container runs with is not enough, and you
should increase shared memory size either with <code>--ipc=host</code> or <code>--shm-size</code> command line options to <code>nvidia-docker run</code>.</p>
<h4>Building the image yourself</h4>
<p><strong>NOTE:</strong> Must be built with a docker version &gt; 18.06</p>
<p>The <code>Dockerfile</code> is supplied to build images with CUDA 11.1 support and cuDNN v8.
You can pass <code>PYTHON_VERSION=x.y</code> make variable to specify which Python version is to be used by Miniconda, or leave it
unset to use the default.</p>
<p>```bash
make -f docker.Makefile</p>
<h1>images are tagged as docker.io/${your_docker_username}/pytorch</h1>
<p>```</p>
<p>You can also pass the <code>CMAKE_VARS="..."</code> environment variable to specify additional CMake variables to be passed to CMake during the build.
See <a href="./setup.py">setup.py</a> for the list of available variables.</p>
<p><code>bash
CMAKE_VARS="BUILD_CAFFE2=ON BUILD_CAFFE2_OPS=ON" make -f docker.Makefile</code></p>
<h3>Building the Documentation</h3>
<p>To build documentation in various formats, you will need <a href="http://www.sphinx-doc.org">Sphinx</a> and the
readthedocs theme.</p>
<p><code>bash
cd docs/
pip install -r requirements.txt</code>
You can then build the documentation by running <code>make &lt;format&gt;</code> from the
<code>docs/</code> folder. Run <code>make</code> to get a list of all available output formats.</p>
<p>If you get a katex error run <code>npm install katex</code>.  If it persists, try
<code>npm install -g katex</code></p>
<blockquote>
<p>Note: if you installed <code>nodejs</code> with a different package manager (e.g.,
<code>conda</code>) then <code>npm</code> will probably install a version of <code>katex</code> that is not
compatible with your version of <code>nodejs</code> and doc builds will fail.
A combination of versions that is known to work is <code>node@6.13.1</code> and
<code>katex@0.13.18</code>. To install the latter with <code>npm</code> you can run
<code>npm install -g katex@0.13.18</code></p>
</blockquote>
<h3>Previous Versions</h3>
<p>Installation instructions and binaries for previous PyTorch versions may be found
on <a href="https://pytorch.org/previous-versions">our website</a>.</p>
<h2>Getting Started</h2>
<p>Three-pointers to get you started:
- <a href="https://pytorch.org/tutorials/">Tutorials: get you started with understanding and using PyTorch</a>
- <a href="https://github.com/pytorch/examples">Examples: easy to understand PyTorch code across all domains</a>
- <a href="https://pytorch.org/docs/">The API Reference</a>
- <a href="https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md">Glossary</a></p>
<h2>Resources</h2>
<ul>
<li><a href="https://pytorch.org/">PyTorch.org</a></li>
<li><a href="https://pytorch.org/tutorials/">PyTorch Tutorials</a></li>
<li><a href="https://github.com/pytorch/examples">PyTorch Examples</a></li>
<li><a href="https://pytorch.org/hub/">PyTorch Models</a></li>
<li><a href="https://www.udacity.com/course/deep-learning-pytorch--ud188">Intro to Deep Learning with PyTorch from Udacity</a></li>
<li><a href="https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229">Intro to Machine Learning with PyTorch from Udacity</a></li>
<li><a href="https://www.coursera.org/learn/deep-neural-networks-with-pytorch">Deep Neural Networks with PyTorch from Coursera</a></li>
<li><a href="https://twitter.com/PyTorch">PyTorch Twitter</a></li>
<li><a href="https://pytorch.org/blog/">PyTorch Blog</a></li>
<li><a href="https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw">PyTorch YouTube</a></li>
</ul>
<h2>Communication</h2>
<ul>
<li>Forums: Discuss implementations, research, etc. https://discuss.pytorch.org</li>
<li>GitHub Issues: Bug reports, feature requests, install issues, RFCs, thoughts, etc.</li>
<li>Slack: The <a href="https://pytorch.slack.com/">PyTorch Slack</a> hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is <a href="https://discuss.pytorch.org">PyTorch Forums</a>. If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1</li>
<li>Newsletter: No-noise, a one-way email newsletter with important announcements about PyTorch. You can sign-up here: https://eepurl.com/cbG0rv</li>
<li>Facebook Page: Important announcements about PyTorch. https://www.facebook.com/pytorch</li>
<li>For brand guidelines, please visit our website at <a href="https://pytorch.org/">pytorch.org</a></li>
</ul>
<h2>Releases and Contributing</h2>
<p>Typically, PyTorch has three major releases a year. Please let us know if you encounter a bug by <a href="https://github.com/pytorch/pytorch/issues">filing an issue</a>.</p>
<p>We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.</p>
<p>If you plan to contribute new features, utility functions, or extensions to the core, please first open an issue and discuss the feature with us.
Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the core in a different direction than you might be aware of.</p>
<p>To learn more about making a contribution to Pytorch, please see our <a href="CONTRIBUTING.md">Contribution page</a>. For more information about PyTorch releases, see <a href="RELEASE.md">Release page</a>.</p>
<h2>The Team</h2>
<p>PyTorch is a community-driven project with several skillful engineers and researchers contributing to it.</p>
<p>PyTorch is currently maintained by <a href="http://soumith.ch">Soumith Chintala</a>, <a href="https://github.com/gchanan">Gregory Chanan</a>, <a href="https://github.com/dzhulgakov">Dmytro Dzhulgakov</a>, <a href="https://github.com/ezyang">Edward Yang</a>, and <a href="https://github.com/malfet">Nikita Shulga</a> with major contributions coming from hundreds of talented individuals in various forms and means.
A non-exhaustive but growing list needs to mention: Trevor Killeen, Sasank Chilamkurthy, Sergey Zagoruyko, Adam Lerer, Francisco Massa, Alykhan Tejani, Luca Antiga, Alban Desmaison, Andreas Koepf, James Bradbury, Zeming Lin, Yuandong Tian, Guillaume Lample, Marat Dukhan, Natalia Gimelshein, Christian Sarofeen, Martin Raison, Edward Yang, Zachary Devito.</p>
<p>Note: This project is unrelated to <a href="https://github.com/hughperkins/pytorch">hughperkins/pytorch</a> with the same name. Hugh is a valuable contributor to the Torch community and has helped with many things Torch and PyTorch.</p>
<h2>License</h2>
<p>PyTorch has a BSD-style license, as found in the <a href="LICENSE">LICENSE</a> file.</p>