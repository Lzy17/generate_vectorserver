<p>The most important things to know:</p>
<p><strong>Don't add a kernel to this folder unless you want it to be
compiled multiple times for different instruction sets.</strong>  Yes,
this folder is named <code>cpu</code>, but that doesn't mean put any old
CPU kernel it.  Only put CPU kernels which need to be compiled
multiple times to take advantage of AVX512/AVX2/SSE instructions, but
only on processors that support them.</p>
<p><strong>Ensure that all implementations in this folder are put in an
anonymous namespace.</strong>  The files in this folder are compiled multiple
times with different headers. It's important that these functions have
internal linkage so that kernels for different architectures don't get
combined during linking.  It's sufficient to label functions "static",
but class methods must be an unnamed namespace to have internal linkage
(since static means something different in the context of classes).</p>
<p><strong>The basic recipe is to define your kernel, and then register
it using DECLARE/REGISTER DISPATCH.</strong>  Writing a kernel requires
three steps:</p>
<ol>
<li>
<p>Declare your dispatch in a header file using
  <code>DECLARE_DISPATCH(fn_type, fnNameImpl);</code>
   where <code>fn_type</code> is the function pointer type of the kernel (e.g.,
   defined as <code>using fn_type = void(*)(Tensor&amp;, const Tensor&amp;)</code>
   and <code>fnNameImpl</code> is the name of your dispatch registry.
   (It doesn't really matter where you  put this declaration.)</p>
</li>
<li>
<p>Define your dispatch in a C++ file that is NOT in the cpu
   directory (dispatch must be defined exactly once) using
   <code>DEFINE_DISPATCH(fnNameImpl)</code> (matching the name of your declaration.)
   Include the header file that declares the dispatch in this C++
   file.  Conventionally, we define the dispatch in the same file
   we will define our native function in.</p>
</li>
<li>
<p>Define a native function which calls into the dispatch using
   <code>fnNameImpl(kCPU, arguments...)</code>, where the arguments are
   the arguments according to the <code>fn_type</code> you defined in the
   declaration.</p>
</li>
<li>
<p>Write your actual kernel (e.g., <code>your_kernel</code>) in the
   cpu directory, and register it to
   the dispatch using <code>REGISTER_DISPATCH(fnNameImpl, &amp;your_kernel)</code>.</p>
</li>
</ol>
<p>There are plenty of existing examples, look at them for more details.</p>
<hr />
<p>TODO: Clarify and add more documentation all around.</p>
<p>All of the <code>*.cpp</code> files in this folder will be compiled under all compiler
flags specified by <code>CPU_CAPABILITY_FLAGS</code> in <code>aten/src/ATen/CMakeLists.txt</code>.</p>
<p>The purpose of this is to allow the compilation with various compiler
flags to enable features such as AVX2 or AVX512 instructions, while using
runtime dispatch, which makes sure only valid instructions will be used on any
given platform.</p>
<p>vec.h provides a generic implementation of vec type that allows
the programmer to write code packing various primitives (such as floats)
within 256bit &amp; 512bits registers. vec defines various operators such as
+ and * and provides functions to allow operations such as max, min, etc.</p>
<p>As an example <code>ReduceOpsKernel.cpp</code> implements a generic <code>kernel_</code> that reduces
an entire array using a given associative binary operation such as +.</p>
<p>More explicitly, calling <code>kernel_</code> with template argument <code>std::plus</code> will cause
it to sum up the entire array into a single value.</p>
<p><code>ReduceOpsKernel.cpp</code> uses the <code>CPU_CAPABILITY_*</code> macros to "know" under which
compiler flags it is currently compiled. This allows the programmer to write
generic code, which will be compiled under multipled compilation settings.</p>
<p><code>../ReduceOps.cpp</code> now includes the header <code>ReduceOpsKernel.h</code>, which contains
a generic definition of <code>sumImplAll</code>. This function allows the user to reduce
over a dimension or all dimensions. The appropriate capability is chosen at
runtime using cpuinfo. If the current platform has AVX2, <code>sumImpl</code> will be set
to <code>sumImplAll&lt;CPUCapability::AVX2&gt;</code>.</p>
<p>At runtime, the following environment variables control which codepath is taken:</p>
<p>x64 options:
ATEN_CPU_CAPABILITY=avx2    # Force AVX2 codepaths to be used
ATEN_CPU_CAPABILITY=avx     # Force AVX codepaths to be used
ATEN_CPU_CAPABILITY=default # Use oldest supported vector instruction set</p>