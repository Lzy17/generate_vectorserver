<p>This folder contains various custom cmake modules for finding libraries and packages. Details about some of them are listed below.</p>
<h3><a href="./FindOpenMP.cmake"><code>FindOpenMP.cmake</code></a></h3>
<p>This is modified from <a href="https://github.com/Kitware/CMake/blob/05a2ca7f87b9ae73f373e9967fde1ee5210e33af/Modules/FindOpenMP.cmake">the file included in CMake 3.13 release</a>, with the following changes:</p>
<ul>
<li>
<p>Replace <code>VERSION_GREATER_EQUAL</code> with <code>NOT ... VERSION_LESS</code> as <code>VERSION_GREATER_EQUAL</code> is not supported in CMake 3.5 (our min supported version).</p>
</li>
<li>
<p>Update the <code>separate_arguments</code> commands to not use <code>NATIVE_COMMAND</code> which is not supported in CMake 3.5 (our min supported version).</p>
</li>
<li>
<p>Make it respect the <code>QUIET</code> flag so that, when it is set, <code>try_compile</code> failures are not reported.</p>
</li>
<li>
<p>For <code>AppleClang</code> compilers, use <code>-Xpreprocessor</code> instead of <code>-Xclang</code> as the later is not documented.</p>
</li>
<li>
<p>For <code>AppleClang</code> compilers, an extra flag option is tried, which is <code>-Xpreprocessor -openmp -I${DIR_OF_omp_h}</code>, where <code>${DIR_OF_omp_h}</code> is a obtained using <code>find_path</code> on <code>omp.h</code> with <code>brew</code>'s default include directory as a hint. Without this, the compiler will complain about missing headers as they are not natively included in Apple's LLVM.</p>
</li>
<li>
<p>For non-GNU compilers, whenever we try a candidate OpenMP flag, first try it with directly linking MKL's <code>libomp</code> if it has one. Otherwise, we may end up linking two <code>libomp</code>s and end up with this nasty error:</p>
</li>
</ul>
<p>```
  OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already
  initialized.</p>
<p>OMP: Hint This means that multiple copies of the OpenMP runtime have been
  linked into the program. That is dangerous, since it can degrade performance
  or cause incorrect results. The best thing to do is to ensure that only a
  single OpenMP runtime is linked into the process, e.g. by avoiding static
  linking of the OpenMP runtime in any library. As an unsafe, unsupported,
  undocumented workaround you can set the environment variable
  KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but
  that may cause crashes or silently produce incorrect results. For more
  information, please see http://openmp.llvm.org/
  ```</p>
<p>See NOTE [ Linking both MKL and OpenMP ] for details.</p>