<p>This folder contains generated sources for the lazy torchscript backend.</p>
<p>The main input file that drives which operators get codegen support for torchscript backend is
<a href="../../../../aten/src/ATen/native/ts_native_functions.yaml">../../../../aten/src/ATen/native/ts_native_functions.yaml</a></p>
<p>The code generator lives at <code>torchgen/gen_lazy_tensor.py</code>.</p>
<p>It is called automatically by the torch autograd codegen (<code>tools/setup_helpers/generate_code.py</code>)
as a part of the build process in OSS builds (CMake/Bazel) and Buck.</p>
<p>External backends (e.g. torch/xla) call <code>gen_lazy_tensor.py</code> directly,
and feed it command line args indicating where the output files should go.</p>
<p>For more information on codegen, see these resources:
* Info about lazy tensor codegen: <a href="../../../../torchgen/gen_lazy_tensor.py">gen_lazy_tensor.py docs</a>
* Lazy TorchScript backend native functions: <a href="../../../../aten/src/ATen/native/ts_native_functions.yaml">ts_native_functions.yaml</a>
* Source of truth for native func definitions <a href="../../../../aten/src/ATen/native/native_functions.yaml">ATen native_functions.yaml</a>
* Info about native functions <a href="../../../../aten/src/ATen/native/README.md">ATen nativefunc README.md</a></p>