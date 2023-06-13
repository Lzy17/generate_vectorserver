<h1>An ATen operator for Caffe2</h1>
<p>ATen is a simple tensor library thats exposes the Tensor operations in Torch
and PyTorch directly in C++17. This library provides a generated wrapper around the ATen API
that makes these functions available in Caffe2 as an operator. It also makes it accessible using the
ToffeeIR.</p>
<h3>Example Usage in Caffe2</h3>
<p>First identify a function in ATen you want to call in Functions.h,
Tensor.h, or Type.h.</p>
<p>We will call the <code>pow</code> operator:</p>
<p><code>static inline Tensor pow(const Tensor &amp; self, Scalar exponent);</code></p>
<p>Now create a Caffe2 operator to call this op. The name of the operator is always <code>"ATen"</code>,
and there is always a string attribute <code>operator</code> that defines which ATen function to call:</p>
<p>```
import numpy as np
from caffe2.python import core, workspace</p>
<h1>create the Caffe2 Op:</h1>
<p>op = core.CreateOperator(
    "ATen",
    ["MyInput"],
    ["MyOutput"],
    operator="pow", exponent=2.0)</p>
<p>```</p>
<p>Each <code>Tensor</code> input becomes an Caffe2 input Blob, and each output becomes a Caffe2 output blob.
Non-tensor inputs such as <code>Scalar exponent</code> become Caffe2 <code>arg</code> attributes.
In the case of <code>Scalar</code> the attributes can be either an integers or floating point numbers.</p>
<p>The op can now be run like any other Caffe2 operator:</p>
<p><code>workspace.FeedBlob("MyInput",np.random.randn(2,3).astype(np.float32))
workspace.RunOperatorOnce(op)
print(workspace.FetchBlob("MyOutput")</code></p>
<p>For methods, the first input is always the <code>this</code> Tensor in C++.
To call methods of ATen's <code>Type</code> objects, you provide an additional string attribute
that determines the type:</p>
<p>```</p>
<h1>create a 2x4 tensor filled with floating point ones</h1>
<p>op = core.CreateOperator(
    "ATen",
    [],
    ["MyOutput"],
    operator="ones", type="Float", size={2,4})
```</p>
<p>Generally ATen operators are polymorphic across input types, and work on both the CPU and CUDA.</p>
<h3>Example Usage via PyTorch Symbolic</h3>
<p>The ATen operator can also be used to define <code>symbolic</code> definitions for PyTorch when an operator is being exported
to ONNX. In this case, the definition of the operator looks the same but is defined using PyTorch's ONNX API:</p>
<p>```
class Add(torch.autograd.Function):</p>
<pre><code>@staticmethod
def symbolic(g, a, b):
    return g.at("add", a, b)

@staticmethod
def forward(ctx, a, b):
    return a + b
</code></pre>
<p>```</p>