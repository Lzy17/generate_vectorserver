<p>This folder contains a number of scripts which are used as
part of the PyTorch build process.  This directory also doubles
as a Python module hierarchy (thus the <code>__init__.py</code>).</p>
<h2>Overview</h2>
<p>Modern infrastructure:</p>
<ul>
<li><a href="autograd">autograd</a> - Code generation for autograd.  This
  includes definitions of all our derivatives.</li>
<li><a href="jit">jit</a> - Code generation for JIT</li>
<li><a href="shared">shared</a> - Generic infrastructure that scripts in
  tools may find useful.</li>
<li><a href="shared/module_loader.py">module_loader.py</a> - Makes it easier
    to import arbitrary Python files in a script, without having to add
    them to the PYTHONPATH first.</li>
</ul>
<p>Build system pieces:</p>
<ul>
<li><a href="setup_helpers">setup_helpers</a> - Helper code for searching for
  third-party dependencies on the user system.</li>
<li><a href="build_pytorch_libs.py">build_pytorch_libs.py</a> - cross-platform script that
  builds all of the constituent libraries of PyTorch,
  but not the PyTorch Python extension itself.</li>
<li><a href="build_libtorch.py">build_libtorch.py</a> - Script for building
  libtorch, a standalone C++ library without Python support.  This
  build script is tested in CI.</li>
</ul>
<p>Developer tools which you might find useful:</p>
<ul>
<li><a href="git_add_generated_dirs.sh">git_add_generated_dirs.sh</a> and
  <a href="git_reset_generated_dirs.sh">git_reset_generated_dirs.sh</a> -
  Use this to force add generated files to your Git index, so that you
  can conveniently run diffs on them when working on code-generation.
  (See also <a href="generated_dirs.txt">generated_dirs.txt</a> which
  specifies the list of directories with generated files.)</li>
</ul>
<p>Important if you want to run on AMD GPU:</p>
<ul>
<li><a href="amd_build">amd_build</a> - HIPify scripts, for transpiling CUDA
  into AMD HIP.  Right now, PyTorch and Caffe2 share logic for how to
  do this transpilation, but have separate entry-points for transpiling
  either PyTorch or Caffe2 code.</li>
<li><a href="amd_build/build_amd.py">build_amd.py</a> - Top-level entry
    point for HIPifying our codebase.</li>
</ul>
<p>Tools which are only situationally useful:</p>
<ul>
<li><a href="docker">docker</a> - Dockerfile for running (but not developing)
  PyTorch, using the official conda binary distribution.  Context:
  https://github.com/pytorch/pytorch/issues/1619</li>
<li><a href="download_mnist.py">download_mnist.py</a> - Download the MNIST
  dataset; this is necessary if you want to run the C++ API tests.</li>
</ul>