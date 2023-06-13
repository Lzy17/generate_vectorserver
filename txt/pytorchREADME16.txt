<p>The quantized folder holds the implementation of the low-level quantized kernel.
The kernels are registered in <code>torch::_ops</code> namespace, and operate on the quantized <code>at::Tensor</code> data type.
You can learn more about the quantized tensors in the <a href="https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor">quantized tensor API wiki</a> page.</p>
<p>This document serves as an entry point for quantized kernel implementation.</p>
<h2>Implementing native quantized ops</h2>
<p>The new quantized ops are almost always located under the <code>ATen/native/quantized/cpu</code> folder. For
the sake of an example, let us implement an element-wise quantized <a href="https://en.wiktionary.org/wiki/XAND">logical XAND</a>
operation under <code>ATen/native/quantized/cpu/qxand.cpp</code>.</p>
<h3>Step 0. Implement the quantized function</h3>
<p>Before writing the quantized kernel and registering it, let us implement a quantized function.
That would assist in any further discussion.
The snippet below shows the implementation of a quantized XAND operator, with the support of all implemented quantized types.</p>
<p>```c++
Tensor quantized_xand(Tensor qa, Tensor qb) {
  // Some type checks for qa and qb should be here...
  Tensor qc;
  double scale = qa.q_scale();
  int64_t zero_point = qa.q_zero_point();</p>
<p>auto iter = TensorIterator::binary_op(qc, qa, qb);</p>
<p>AT_DISPATCH_QINT_TYPES(qa.scalar_type(), "quantized_xand", <a href="">&amp;</a> {
    Tensor qc = at::<em>empty_affine_quantized(
        qa.sizes(), at::device(kCPU).dtype(SCALAR_TYPE), scale, zero_point);
    cpu_kernel(iter, <a href="scalar_t a_value, scalar_t b_value">&amp;</a> -&gt; scalar_t {
      return scalar_t(a_value.val</em> &amp; b_value.val_);
    });
  });
  return qc;
}
```</p>
<p>The code above is fairly straight-forward:
It takes two quantized tensors <code>qa</code> and <code>qb</code>, and uses <code>binary_kernel</code> to produce a quantized tensor <code>qc</code>.
We also use the <a href="https://caffe2.ai/doxygen-c/html/structat_1_1_tensor_iterator.html"><code>TensorIterator</code></a> in this example.
The only part that that requires explicit explanation is the <code>AT_DISPATCH_QINT_TYPES</code>.
This macro makes sure that the underlying code works with all quantized types.
It provides several useful "aliases":</p>
<ul>
<li><code>SCALAR_TYPE</code> -- <code>ScalarType</code> of the quantized tensor (e.g. <code>kQInt8</code>)</li>
<li><code>scalar_t</code> -- quantized data type (dtype, e.g. <code>qint8</code>)</li>
<li><code>underlying_t</code> -- underlying POD data type (dtype, e.g. <code>int8_t</code>)</li>
</ul>
<p>The macro takes three arguments:</p>
<ol>
<li>Quantized data type. This will define what the "aliases" are.
In the example above, the resulting tensor will be the same as the <code>qa.scalar_type()</code>.</li>
<li>Function name. This argument is currently used for error reporting.</li>
<li>Implementation lambda. The main implementation should sit in the body of this lambda.
it should also use the aliases for the quantized data types instead of the explicit data types.</li>
</ol>
<h3>Step 1. Define the schema</h3>
<p>Update <code>aten/src/ATen/native/quantized/library.cpp</code> and add
a <code>def</code> for your new operator:</p>
<p><code>c++
TORCH_LIBRARY(quantized, m) {
  // ... the existing definitions ...
  m.def("quantized::xand(Tensor qa, Tensor qb) -&gt; Tensor");
}</code></p>
<p>Def takes a <strong>function schema string</strong>: This schema describes the usage of the op.
In the example above the schema is <code>"quantized::xand(Tensor qa, Tensor qb) -&gt; Tensor"</code>.
This translates to <code>torch._ops.ops.quantized.xand</code> function in Python of the appropriate signature.</p>
<h3>Step 2. Register the implementation</h3>
<p>The registration is done using <code>TORCH_LIBRARY_IMPL</code>.</p>
<p><code>c++
TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl("xand", TORCH_FN(quantized_xand));
}</code></p>
<h3>Step 2b. [Optional] Registering the operation with the <code>native_functions.yaml</code></h3>
<p>In some cases, if the signature of the quantized function and its non-quantized counterpart are the same, it is worth adding it to the <code>ATen/native/native_functions.yaml</code>.
A detailed explanation on this file can be found <a href="https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/README.md">here</a>.</p>
<p><strong>If adding a new entry to the <code>native_functions.yaml</code>:</strong></p>
<p><code>yaml
- func: quantized_xand(Tensor qa, Tensor qb) -&gt; Tensor
  dispatch:
    QuantizedCPU: quantized_xand</code></p>
<p><strong>If adding to an existing entry in the <code>native_functions.yaml</code>:</strong></p>
<p>If you find an entry in the yaml file, and would like to add a quantized kernel to it, you can just add a new dispatch entry for it.
For example, let's assume there existed a <code>xand</code> function in the YAML file.
In that case, modification would look as:</p>
<p><code>yaml
- func: xand(Tensor a, Tensor b) -&gt; Tensor
  dispatch:
    CPU: _xand_cpu     # Assume this existed
    CUDA: _xand_cuda   # Assume this existed
    QuantizedCPU: quantized_xand</code></p>
<h3>Putting it all together</h3>
<p>The final file <code>ATen/native/quantized/cpu/qxand.cpp</code> would look as follows</p>
<p>```c++</p>
<h1>include <ATen/ATen.h></h1>
<h1>include <ATen/NativeFunctions.h> // Need that for the <code>native_functions.yaml</code></h1>
<h1>include <ATen/core/Type.h></h1>
<h1>include <torch/library.h></h1>
<h1>include <ATen/native/TensorIterator.h></h1>
<h1>include <ATen/native/cpu/Loops.h></h1>
<p>namespace at {
  namespace native {
  Tensor quantized_xand(Tensor qa, Tensor qb) {
    // The awesome op implementation...
    return qc;
  }</p>
<p>TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
    m.impl("xand", TORCH_FN(quantized_xand));
  }
}}  // namespace at::native
```</p>
<h3>Step 3. Administrative stuff</h3>
<p>Before the op can be used, it needs to be compiled.
If the op is placed under <code>native/quantized/cpu</code>, this already done for you.
However, if the location is changed, two files must be notified:</p>
<ul>
<li><em><code>caffe2/aten/TARGETS</code></em> -- You can follow the same example, and add your path in somewhere in that file. Notice in this file we places the path to the quantized source files:
```bash
ATEN_NATIVE_CPP = glob([</li>
</ul>
<h1>...</h1>
<p>"src/ATen/native/quantized/*<em>/</em>.cpp",
])
```</p>
<ul>
<li><em><code>caffe2/aten/src/ATen/CMakeLists.txt</code></em> -- Again, following the example, you must add your paths.
The current quantization paths are added as</li>
</ul>
<p><code>bash
FILE(GLOB native_quantized_cpp
          "native/quantized/*.cpp"
          "native/quantized/cpu/*.cpp")</code></p>
<h2>Using quantized ops</h2>
<h3>Python</h3>
<p>Usage in Python is pretty easy.
To implement the python quantized function using our kernel, you can do the following</p>
<p>```python
from torch._ops import ops</p>
<p>def quantized_xand(qa, qb):</p>
<h1>Notice the schema changed from <code>quantized::xand</code> to <code>quantized.xand</code></h1>
<p>return ops.quantized.xand(qa, qb)
```</p>
<p><strong>Note:</strong> If writing new pytorch functions that use quantized kernels,
it is strongly encouraged to place them in the <code>torch/ao/nn/quantized/functional.py</code>.</p>
<h3>C++</h3>
<p>You should not need to use the registered kernels in C++.
Although <strong>officially not supported</strong>, you can use the following</p>
<p><code>c++
  Tensor quantized_xand(Tensor qa, Tensor qb) {
    static const c10::OperatorHandle op = c10::Dispatcher::singleton().findSchema({"quantized::xand", ""}).value();
    return op.call&lt;Tensor, Tensor, Tensor&gt;(qa, qb);
  }</code></p>