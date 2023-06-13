<p>c10/cuda is a core library with CUDA functionality.  It is distinguished
from c10 in that it links against the CUDA library, but like c10 it doesn't
contain any kernels, and consists solely of core functionality that is generally
useful when writing CUDA code; for example, C++ wrappers for the CUDA C API.</p>
<p><strong>Important notes for developers.</strong> If you want to add files or functionality
to this folder, TAKE NOTE.  The code in this folder is very special,
because on our AMD GPU build, we transpile it into c10/hip to provide a
ROCm environment.  Thus, if you write:</p>
<p>```
// c10/cuda/CUDAFoo.h
namespace c10 { namespace cuda {</p>
<p>void my_func();</p>
<p>}}
```</p>
<p>this will get transpiled into:</p>
<p>```
// c10/hip/HIPFoo.h
namespace c10 { namespace hip {</p>
<p>void my_func();</p>
<p>}}
```</p>
<p>Thus, if you add new functionality to c10, you must also update <code>C10_MAPPINGS</code>
<code>torch/utils/hipify/cuda_to_hip_mappings.py</code> to transpile
occurrences of <code>cuda::my_func</code> to <code>hip::my_func</code>.  (At the moment,
we do NOT have a catch all <code>cuda::</code> to <code>hip::</code> namespace conversion,
as not all <code>cuda</code> namespaces are converted to <code>hip::</code>, even though
c10's are.)</p>
<p>Transpilation inside this folder is controlled by <code>CAFFE2_SPECIFIC_MAPPINGS</code>
(oddly enough.)  <code>C10_MAPPINGS</code> apply to ALL source files.</p>
<p>If you add a new directory to this folder, you MUST update both
c10/cuda/CMakeLists.txt and c10/hip/CMakeLists.txt</p>