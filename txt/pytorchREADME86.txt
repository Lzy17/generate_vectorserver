<h1>torch.onnx</h1>
<p>Torch-&gt;ONNX converter / exporter.</p>
<ul>
<li>User-facing docs: https://pytorch.org/docs/master/onnx.html</li>
<li>Developer docs: https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter</li>
</ul>
<blockquote>
<p>Read the following if you are contributing to <code>torch.onnx</code></p>
</blockquote>
<h2>Symbolic functions Opsets</h2>
<p>Opset 9 is the base version. It is selected as the base version because</p>
<ol>
<li>It is the first opset version supported by PyTorch export.</li>
<li>Opset 9 is more robust than previous opset versions. Opset versions like 7/8 have limitations
    that certain basic operators cannot be expressed in ONNX. Instead of basing on these limitations,
    we chose to handle them as special cases separately.</li>
</ol>
<p>Backward support for opset versions beyond opset 7 is not in our roadmap.</p>
<p>For opset versions other than 9, by default they will inherit the symbolic functions defined in
symbolic_opset9.py.</p>
<p>To extend support for updated operators in different opset versions on top of opset 9,
simply add the updated symbolic functions in the respective symbolic_opset{version}.py file.
Checkout topk in symbolic_opset10.py, and upsample_nearest2d in symbolic_opset8.py for example.</p>
<h2>Editing Symbolic Files</h2>
<ul>
<li>Use the internal <code>registration.onnx_symbolic</code> decorator to register a new symbolic function. Search for <code>def reshape(g, self, shape):</code> to see an example.</li>
<li>Parameter names must <em>exactly</em> match the names in
  aten/src/ATen/native/native_functions.yaml, because
  dispatch is done with keyword arguments.</li>
<li>Looking for inplace ops? They're detected by
  <code>_jit_pass_onnx_remove_inplace_ops_for_onnx</code>, and
  transparently dispatched to their non inplace versions in
  "run_symbolic_function". See Note <a href="#export-inplace">Export inplace</a></li>
<li>Required: Annotate new symbolic functions with type annotations and decorate
  with <code>@_beartype.beartype</code> to enable runtime type checking.
  <code>@_beartype.beartype</code> should typically be the closest to the function to
  ensure proper type checking.</li>
</ul>
<h3>A note on Tensor types</h3>
<p>In general, we should avoid depending on the type of Tensor Values contained
within the trace graph. However, this is sometimes unavoidable (due to ONNX
spec requirements, etc). The TensorType object has accessors for these properties that return the property if it is statically known and return nullopt otherwise.</p>
<p>In general, we should prefer to rely on the least specific information possible.
For example, not relying on tensor properties at all is better than relying
on the number of dimensions which is better than relying on
concrete shapes. Doing so will make the export symbolics
more robust to different graphs.</p>
<h3>Extra context for symbolic functions</h3>
<p>The first argument of a symbolic function is always a <code>GraphContext</code> object.</p>
<p><code>GraphContext</code> contains all methods defined in a <code>torch.Graph</code> object and context
for the symbolic function.</p>
<p>In general, symbolic functions only require inputs and attributes to
the original node. An example of a symbolic function needing context is
<code>prim::Loop</code>. It needs access to the sub-block of the original node.</p>
<h3>Export inplace</h3>
<p>It would be better for us to export inplace annotations,
than to not export them, since it is useful information that can
help the target of an ONNX export export more efficiently. However,
ONNX doesn't currently formalize inplace. Fortunately, it's sound to drop
inplace annotations, but we are losing information this way.</p>
<h3>Pointwise by scalar</h3>
<p>What happens if you add a tensor with a constant (e.g., x + 2)?  There are
some moving parts to implementing the ONNX translation in this case:</p>
<ul>
<li>
<p>By the time we get the scalar in a symbolic function here, it is no longer a
  Python long/float, but a PyTorch tensor with <code>numel == 1</code> (eventually, we want
  it to be a zero dim tensor but this change has not happened yet.) However, the
  type of this scalar is <em>exactly</em> what the user wrote in Python, which may not
  match the tensor it is being added to. PyTorch will do implicit conversions on
  scalars; however, ONNX will not, so we must do the conversion ourselves. This
  is what <code>symbolic_helper._if_scalar_type_as()</code> and
  <code>_jit_pass_onnx_scalar_type_analysis</code> does.</p>
</li>
<li>
<p>Dispatch to these functions takes advantage an outrageous coincidence
    between the tensor and scalar name.  When we add two tensors together,
    you get the dispatch:</p>
<p>add(<em>[self, other], </em>*{"alpha": alpha})</p>
<p>When you add a tensor and a scalar, you get the dispatch:</p>
<p>add(<em>[self], </em>*{"other": other, "alpha": alpha})</p>
<p>By having the argument name line up with the name of the scalar attribute
if it exists, we can write a single function for both overloads.</p>
</li>
</ul>