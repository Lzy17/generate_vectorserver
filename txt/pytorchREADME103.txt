<h1>Pytorch - oneDNN Graph API Bridge</h1>
<p>This is a PyTorch JIT graph fuser based on <a href="https://spec.oneapi.io/onednn-graph/latest/programming_model.html">oneDNN Graph API</a>, which provides a flexible API for aggressive fusion. Float &amp; BFloat16 inference is supported. However, BFloat16 only performs well on Intel Xeon Cooper Lake platform &amp; beyond, as they have native BFloat16 support. Also, currently, PyTorch has divergent AMP support in JIT &amp; eager modes, so one should disable JIT AMP support &amp; leverage eager mode AMP support to use BFloat16. Please refer to the BFloat16 example below.</p>
<p>Currently, speedup is achieved only for static shapes, although we'd soon add dynamic-shape support. When oneDNN Graph is enabled, weights are cached, as they're constant during inference.</p>
<h2>Graph Optimization</h2>
<p>We have registered optimization passes in the custom pre-passes set of PyTorch:</p>
<ol>
<li>
<p>Alias and mutation reduction</p>
<p>The operators of oneDNN graph are pure functional while PyTorch has operators in in-place forms or create views for buffer sharing.
Due to the semantic gaps between the backend operators and the PyTorch operators, we have a pass to reduce mutation with best effort at the beginning.</p>
</li>
<li>
<p>Graph passing</p>
<p>With a PyTorch TorchScript graph, the integration maps PyTorch operators on the graph to the corresponding oneDNN Graph operators to form a backend graph.</p>
</li>
<li>
<p>Partitioning</p>
<p>The backend selects regions to be fused in the graph and returns a list of partitions. Each partition corresponds to a set of fused operators.</p>
</li>
<li>
<p>Graph rewriting</p>
<p>The original PyTorch JIT graph will be re-written based on the partitions returned from the backend. The operators in one partition will be grouped together to form a JIT operator, referred to as a oneDNN Graph fusion group.</p>
</li>
<li>
<p>Layout propagation</p>
<p>This pass is to eliminate unnecessary layout conversions at partition boundaries. We set different formats to the output of a partition so that the backend could perform layout conversion internally. When <code>ANY</code> is set, the layout at boundaries will be fully decided by the backend. Otherwise, the backend should follow the layout set by PyTorch. Currently, we set <code>ANY</code> layout for a tensor that's an output of a oneDNN Graph partition, and an input to another.</p>
</li>
</ol>
<h2>Graph Executor</h2>
<p>During runtime execution of a (re-written) PyTorch JIT graph, oneDNN graph partitions will be dispatched to the oneDNN graph JIT variadic Operator.
Inside the oneDNN graph JIT Op, input PyTorch tensors of each partition will be mapped to oneDNN graph tensors. The partition will then be <a href="https://spec.oneapi.io/onednn-graph/latest/programming_model.html#partition">compiled</a> and <a href="https://spec.oneapi.io/onednn-graph/latest/programming_model.html#compiled-partition">executed</a>. The output oneDNN graph tensor will be mapped back to PyTorch tensors to be fed to the next operator on the PyTorch JIT graph.</p>
<h2>Tests</h2>
<p><code>bash
pytest test/test_jit_llga_fuser.py</code></p>
<h2>Quick Start</h2>
<p>A simple cascaded Conv-Relu example is provided in test. Please consider enabling log outputs to familiarize yourself with the whole pipeline:</p>
<p><strong>Mutation Removal -&gt; Prepare Binary -&gt; Defer Size Check -&gt; Graph Fuser -&gt; Layout Propagation -&gt; Type Guard -&gt; Kernel Execution</strong></p>
<p>oneDNN Graph was formerly known as LLGA (Low Level Graph API),
and thus LLGA in the codebase corresponds to oneDNN Graph.</p>
<p><code>bash
DNNL_VERBOSE=1 PYTORCH_JIT_LOG_LEVEL="&gt;&gt;graph_helper:&gt;&gt;graph_fuser:&gt;&gt;kernel:&gt;&gt;interface" python -u test/test_jit_llga_fuser.py -k test_conv2d_eltwise</code></p>
<h2>Codebase structure</h2>
<p>Most of the source code is placed in</p>
<p><code>bash
torch/csrc/jit/codegen/onednn/*</code></p>
<p>Tensor related code is located at</p>
<p><code>bash
torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h
torch/csrc/jit/codegen/onednn/LlgaTensorImpl.cpp</code></p>
<p>CMake files where bridge code is included:</p>
<p><code>bash
caffe2/CMakeLists.txt</code></p>
<p>CMake files where oneDNN Graph submodule are included:</p>
<p><code>bash
third_party/ideep/mkl-dnn
cmake/public/mkldnn.cmake
cmake/Modules/FindMKLDNN.cmake
cmake/Dependencies.cmake</code></p>
<p>To map another op to oneDNN Graph, you should add an entry for it in in createOperator in torch/csrc/jit/codegen/onednn/graph_helper.cpp.
If it has an inplace variant, you should add it in the lambda being passed to RemoveTensorMutation in
torch/csrc/jit/codegen/onednn/interface.cpp. You might also want to add it to canFuseNode in torch/csrc/jit/codegen/onednn/register_interface.cpp.</p>
<h2>Example with Float</h2>
<p>```python</p>
<h1>enable oneDNN graph fusion globally</h1>
<p>torch.jit.enable_onednn_fusion(True)</p>
<h1>define the model</h1>
<p>def MyModel(torch.nn.Module):
    ...</p>
<h1>construct the model</h1>
<p>model = MyModel(…)
with torch.no_grad():
    model.eval()
    model = torch.jit.trace(model, torch.rand(args.batch_size, 3, 224, 224))</p>
<h1>run the model</h1>
<p>with torch.no_grad():
    # oneDNN graph fusion will be triggered during runtime
    output = model(images)
```</p>
<h2>Example with BFloat16</h2>
<p>```python</p>
<h1>Assuming we have a model of the name 'model'</h1>
<p>example_input = torch.rand(1, 3, 224, 224)</p>
<h1>enable oneDNN Graph</h1>
<p>torch.jit.enable_onednn_fusion(True)</p>
<h1>Disable AMP for JIT</h1>
<p>torch._C._jit_set_autocast_mode(False)
with torch.no_grad(), torch.cpu.amp.autocast():
    model = torch.jit.trace(model, (example_input))
    model = torch.jit.freeze(model)
     # 2 warm-ups (2 for tracing/scripting with an example, 3 without an example)
    model(example_input)
    model(example_input)</p>
<pre><code># speedup would be observed in subsequent runs.
model(example_input)
</code></pre>
<p>```</p>