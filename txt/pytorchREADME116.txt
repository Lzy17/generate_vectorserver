<h1>nvFuser Python Frontend</h1>
<p>This frontend allows for a user to describe the set of operations for nvFuser to fuse via 1 or more kernels.  This frontend is intended to be an integration point with PyTorch or standalone applications.</p>
<h1>Usage</h1>
<h2>Example 1 - Define and Execute a Fusion</h2>
<p>```python
import torch
from nvfuser._C import Fusion, FusionDefinition, DataType</p>
<p>fs = Fusion()
with FusionDefinition(fs) as fd :
    t0 = fd.define_tensor(symbolic_sizes=[-1, 1, -1],
                          contiguous=[True, True, True],
                          dtype=DataType.Float)
    t1 = fd.define_tensor(3)
    c0 = fd.define_constant(3.0)</p>
<pre><code>t2 = fd.ops.add(t0, t1)
t3 = fd.ops.mul(t2, c0)
t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

fd.add_output(t4)
</code></pre>
<p>input1 = torch.ones(2, 1, 8, device='cuda')
input2 = torch.ones(2, 4, 8, device='cuda')</p>
<p>nvf_out = fs.execute([input1, input2])[0]
```</p>
<h2>Example 2 - Lookup and Execute a <code>Fusion</code> Based on Id</h2>
<p>```python
fid = 0
fs = Fusion(fid)</p>
<p>input1 = torch.ones(2, 1, 8, device='cuda')
input2 = torch.ones(2, 4, 8, device='cuda')</p>
<p>nvf_out = fs.execute([input1, input2])[0]
```</p>
<h2>Components</h2>
<h3><code>Fusion</code> - Represents a Fusion</h3>
<h4><code>Fusion</code> Methods</h4>
<ul>
<li><code>defined()</code>: Allows you to query if the <code>Fusion</code> is already defined and can be executed.</li>
<li><code>execute([inputs])</code>:  Allows you to execute the currently defined fusion with a list of given inputs and returns a list of tensors.</li>
<li><code>id()</code>: Returns the fusion id for a given <code>Fusion</code>.</li>
<li><code>print()</code>: Prints the low level IR for the currently defined fusion.</li>
</ul>
<h3><code>FusionDefinition</code> Context Manager - Interface for Defining Fusions</h3>
<h4>Defining Input Tensors</h4>
<p><em>All intermediate tensors are created by operations.  Constant tensors do not exist.</em></p>
<p>There are 3 ways to define tensors that will be enumerated below.</p>
<h5>1.) Defining tensors by the number of input dimensions only</h5>
<p>This interface tells nvFuser that the tensor has a given number of symbolic dimensions that are not necessarily contiguous in memory.  The user also has the ability to specify a data type.  The default type is <code>Float</code>.
<code>python
t0 = fd.define_tensor(3)
t1 = fd.define_tensor(3, DataType.Half)</code></p>
<h5>2.) Defining tensors by a list of concrete sizes and a list of strides</h5>
<p>The <code>sizes</code> parameter defines the number of dimensions and the size of each dimension.  The <code>strides</code> parameter has to have the same number of dimensions as the <code>sizes</code> parameter.
nvFuser translates the concrete sizes and strides into symbolic sizes and contiguity information that can be directly defined via the next way to define tensors.  This allows the user to directly take a Pytorch defined tensor and query its sizes and strides in order to apply them in the definition.
<code>python
t0 = fd.define_tensor(sizes=[2, 4, 6], strides=[24, 6, 1], dtype=DataType.Half)</code></p>
<h5>3.) Defining tensors by a list of symbolic sizes and a list of contiguity information</h5>
<p>The list of symbolic sizes defines the number of dimensions and <code>-1</code> is given for each dimension unless it is a broadcast dimension that is defined with a <code>1</code>.  The contiguity information is viewed from right to left.  A <code>True</code> definition indicates the current dimension is contiguous with the dimension to its right.</p>
<p><code>python
t0 = fd.define_tensor(symbolic_sizes=[-1, 1, -1], contiguous=[True, True, True], dtype=DataType.Float)</code></p>
<h4>Defining Input Scalars</h4>
<p><em>All intermediate scalars, except for constants, are created by operations.</em></p>
<p>The only thing the user has to define for a scalar is its type.</p>
<p><code>python
s0 = fd.define_scalar(dtype=DataType.Half)</code></p>
<h4>Defining Constant Scalars</h4>
<p>Constants can be of types: <code>Bool</code>, <code>ComplexDouble</code>, <code>Double</code>, or <code>Int</code>.  The definition only takes a constant and the type is inferred by the constant.</p>
<p><code>python
c0 = fd.define_constant(3.0)</code></p>
<h4>Defining Operations</h4>
<p>Operators are added with the following notation:
<code>python
output = fd.ops.foo(arg1, ... )</code>
You can see a supported list of operations with the following query:
<code>python
python -c "from nvfuser._C import FusionDefinition; help(FusionDefinition.Operators)"</code></p>
<h4>Notating Outputs</h4>
<p>The <code>FusionDefinition</code> <code>add_output</code> method is used to indicate an intermediate is an output to the fusion.</p>
<p>```python
add_output(output: Tensor)</p>
<h1>or</h1>
<p>add_output(output: Scalar)
```</p>
<h1>Debug Information</h1>
<p><strong>Query a list of supported operations:</strong>
<code>python
python -c "from nvfuser._C import FusionDefinition; help(FusionDefinition.Operators)"</code>
<strong>View the fusion definitions that are executed by setting an environment variable:</strong>
<code>python
export PYTORCH_NVFUSER_DUMP=python_definition</code>
Example Output:
<code>python
def nvfuser_fusion_id0(fd : FusionDefinition) -&gt; None :
    T0 = fd.define_tensor(symbolic_sizes=[-1, 1, -1], contiguous=[True, True, True], dtype=DataType.Float)
    T1 = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[False, False, False], dtype=DataType.Float)
    S2 = fd.define_constant(3.00000)
    T3 = fd.ops.add(T0, T1)
    T4 = fd.ops.mul(T3, S2)
    T5 = fd.ops.sum(T4, axes=[-1], keepdim=False, dtype=DataType.Float)
    fd.add_output(T5)</code></p>