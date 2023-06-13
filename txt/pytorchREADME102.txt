<h1>NVFuser - A Fusion Code Generator for NVIDIA GPUs</h1>
<p><em>NVFuser is integrated as a backend for TorchScript's Profiling Graph Executor. NVFuser is the default fuser for NVIDIA GPUs.</em></p>
<h2>Simple knobs to change fusion behavior</h2>
<ol>
<li>
<p>Allow single node fusion <code>torch._C._jit_set_nvfuser_single_node_mode(True)</code>
Fusion group is only created when two or more compatible ops are grouped together. Turn on single node fusion would allow fusion pass to create fusion group with a single node, this is very handy for testing and could be useful when single node generated kernel out-performs native cuda kernels in framework.</p>
</li>
<li>
<p>Allow horizontal fusion <code>torch._C._jit_set_nvfuser_horizontal_mode(True)</code>
Fusion pass fuses producer to consumer, horizontal mode allows sibling nodes that shared tensor input to be fused together. This could save input memory bandwidth.</p>
</li>
<li>
<p>Turn off guard for fusion <code>torch._C._jit_set_nvfuser_guard_mode(False)</code>
This disables the runtime check on fusion group pre-assumptions (tensor meta information / constant inputs / profiled constants), this really is only used for testing as we want to ensure generated kernels are indeed tested and you should avoid using this in training scripts.</p>
</li>
<li>
<p>Turn off fusion for certain node kinds <code>torch._C._jit_set_nvfuser_skip_node_kind("aten::add", True)</code>
This disables fusion for certain nodes, but allows other nodes to continue being fused. The first parameter is the node kind, and the second parameter is whether to toggle the node on or off in fusion.</p>
</li>
</ol>
<h2>Fusion Debugging</h2>
<p>Given the following script as an example</p>
<p>```
import torch</p>
<p>def forward(x):
    o = x + 1.0
    o = o.relu()
    return o</p>
<p>shape = (2, 32, 128, 512)
input = torch.rand(*shape).cuda()
t = torch.jit.script(forward)</p>
<p>with torch.jit.fuser("fuser2"):
    for k in range(4):
        o = t(input)
```</p>
<h3>TorchScript Based Debugging</h3>
<h4>1. TorchScript IR Graph</h4>
<h5>Usage</h5>
<p>Two easy ways to checkout fusion for graph: The first one is to print out graph in python script after a few runs (for optimization to kick in).</p>
<p><code>print(t.graph_for(input))</code></p>
<p>The second way is to turn on graph dumping in profiling executor via command line below:</p>
<p><code>PYTORCH_JIT_LOG_LEVEL="profiling_graph_executor_impl" python &lt;your pytorch script&gt;</code></p>
<h5>Example Output</h5>
<p>Graph print out is straight forward and you should look for <code>prim::CudaFusionGroup_X</code> for fused kernels. While profiling executor dumps many things, but the most important part is <code>Optimized Graph</code>. In this example, it shows a Fusion Group, which is an indication that fusion is happening and you should be expecting fused kernel!</p>
<p><code>Optimized Graph:
  graph(%x.1 : Tensor):
    %12 : bool = prim::CudaFusionGuard[types=[Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0)]](%x.1)
    %11 : Tensor = prim::If(%12)
      block0():
        %o.8 : Tensor = prim::CudaFusionGroup_0[cache_id=0](%x.1)
        -&gt; (%o.8)
      block1():
        %18 : Function = prim::Constant[name="fallback_function", fallback=1]()
        %19 : (Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0)) = prim::CallFunction(%18, %x.1)
        %20 : Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0) = prim::TupleUnpack(%19)
        -&gt; (%20)
    return (%11)
  with prim::CudaFusionGroup_0 = graph(%2 : Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0)):
    %4 : int = prim::Constant[value=1]()
    %3 : float = prim::Constant[value=1.]() # test.py:6:12
    %o.1 : Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0) = aten::add(%2, %3, %4) # test.py:6:8
    %o.5 : Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0) = aten::relu(%o.1) # test.py:7:8
    return (%o.5)</code></p>
<p>Note that one thing that could prevents fusion when you are running training is autodiff. Fusion pass only runs within <code>prim::DifferentiableGraph</code>, so the first thing you should check is to that targetted ops are within differentiable graph subgraphs.
Graph dump could be quite confusing to look at, since it naively dumps all graphs executed by profiling executor and differentiable graphs are executed via a nested graph executor. So for each graph, you might see a few segmented <code>Optimized Graph</code> where each corresponds to a differentiable node in the original graph.</p>
<h4>2. Cuda Fusion Graphs</h4>
<h5>Usage</h5>
<p>Cuda fusion dump gives the input and output graph to fusion pass. This is a good place to check fusion pass logic.</p>
<p><code>PYTORCH_JIT_LOG_LEVEL="graph_fuser" python &lt;your pytorch script&gt;</code></p>
<h5>Example Output</h5>
<p>Running the same script above, in the log, you should be looking for two graphs <code>Before Fusion</code> shows the subgraph where fusion pass runs on; <code>Before Compilation</code> shows the graph sent to codegen backend, where each <code>CudaFusionGroup</code> will trigger codegen runtime system to generate kernel(s) to execute the subgraph.</p>
<p>```
  Before Fusion:
  graph(%x.1 : Tensor):
    %2 : float = prim::Constant<a href="">value=1.</a>
    %1 : int = prim::Constant<a href="">value=1</a>
    %3 : Tensor = prim::profile<a href="%x.1">profiled_type=Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0)</a>
    %o.10 : Tensor = aten::add(%3, %2, %1) # test.py:6:8
    %5 : Tensor = prim::profile<a href="%o.10">profiled_type=Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0)</a>
    %o.7 : Tensor = aten::relu(%5) # test.py:7:8
    %7 : Tensor = prim::profile<a href="%o.7">profiled_type=Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0)</a>
    %8 : Tensor = prim::profile<a href="%o.7">profiled_type=Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0)</a>
    return (%7, %8)</p>
<p>Before Compilation:
  graph(%x.1 : Tensor):
    %13 : bool = prim::CudaFusionGuard<a href="%x.1">types=[Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0)]</a>
    %12 : Tensor = prim::If(%13)
      block0():
        %o.11 : Tensor = prim::CudaFusionGroup_0(%x.1)
        -&gt; (%o.11)
      block1():
        %o.7 : Tensor = prim::FallbackGraph_1(%x.1)
        -&gt; (%o.7)
    return (%12, %12)
  with prim::CudaFusionGroup_0 = graph(%2 : Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0)):
    %4 : int = prim::Constant<a href="">value=1</a>
    %3 : float = prim::Constant<a href="">value=1.</a>
    %o.10 : Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0) = aten::add(%2, %3, %4) # test.py:6:8
    %o.7 : Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0) = aten::relu(%o.10) # test.py:7:8
    return (%o.7)
  with prim::FallbackGraph_1 = graph(%x.1 : Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0)):
    %1 : int = prim::Constant<a href="">value=1</a>
    %2 : float = prim::Constant<a href="">value=1.</a>
    %o.10 : Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0) = aten::add(%x.1, %2, %1) # test.py:6:8
    %o.7 : Float(2, 32, 128, 512, strides=[2097152, 65536, 512, 1], requires_grad=0, device=cuda:0) = aten::relu(%o.10) # test.py:7:8
    return (%o.7)
```</p>
<h3>General ideas of debug no-fusion</h3>
<p>Currently there we have a few consumers that utilizes nvfuser via lowering computations to TorchScript and executing that through a ProfilingExecutor.</p>
<p>Without going into too much details about how the integration is done, a few notes on debugging no-fusion on ProfilingExecutor:</p>
<ol>
<li>
<p>Run TorchScript module multiple times (5 could be a lucky number) to enable fusion.
    Because ProfilingExecutor takes the first (few) runs for profiling, later optimization (including the fusion pass the enables nvfuser) relies on profiling information to run, so your initial runs are not going to trigger fused kernels.
    Note that the number of profiling runs is dependent on your model.</p>
</li>
<li>
<p>Fused kernel should show up in TorchScript IR as <code>prim::CudaFusionGroup</code>. You can look at your TorchScript optimized graph to see if fusion is happening <code>jit_model.graph_for(*inputs)</code>.</p>
</li>
<li>
<p>If your scripted model has inputs requiring gradient, fusion is only happening for graphs inside <code>prim::DifferentiableGraph</code>.
    There are many reasons why your graph is not autodiff-able. Take a look at <code>/torch/csrc/jit/runtime/symbolic_scripts.cpp</code>, which lists all autodiff-able ops (note that this is a different list from autograd-supported ops). There's also a threshold where tiny autodiff graph are inlined/reverted, which could be disabled via <code>torch._C._debug_set_autodiff_subgraph_inlining(False)</code>.</p>
</li>
</ol>
<h3>General ideas of debug nvfuser mal-functioning</h3>
<p>Assuming we have ProfilingExecutor things worked out properly, that is, you see a region that's supposed to be fused but did not ended up in a fused kernel, here's ways to dig deeper:</p>
<ol>
<li>
<p>Dump fusion pass result:
    <code>PYTORCH_JIT_LOG_LEVEL=graph_fuser python your_script.py &amp;&gt; log</code></p>
<p>Looks for graph dumped with <code>Before Fusion</code> &amp; <code>Before Compilation</code>, which shows the portion of graph where fusion pass runs on and the result of fusion (<code>CudaFusionGroup</code>).</p>
</li>
<li>
<p>Check out which ops are not fused and roughly why:
    <code>PYTORCH_JIT_LOG_LEVEL="&gt;partition:graph_fuser" python your_script.py &amp;&gt; log</code></p>
<p>Enabling GRAPH_UPDATE from partition.cpp dumps a log when a given node is rejected by fusion.</p>
</li>
<li>
<p>Disabling FALLBACK path:
    If you see a warning where a FALLBACK path has been taken while executing your model with nvfuser enabled, it's indicating that either codegen or fusion pass has failed unexpectedly. This is likely to cause regression on model performance, even though it's still functionally correct. We recommend to disable FALLBACK path, so error would be reported properly to open an informative issue.</p>
<p><code>PYTORCH_NVFUSER_DISABLE=fallback python your_script.py &amp;&gt; log</code></p>
</li>
<li>
<p>Pin point kernel/fusion pattern that's causing error:
    With a larger model that includes multiple fusion patterns, it could be tricky to figure out which exact fusion is causing FALLBACK and build up a minimal python repro.
    One quick thing to try is to run the example with a few knobs turned on:</p>
<p><code>PYTORCH_NVFUSER_DISABLE=fallback \
PYTORCH_JIT_LOG_LEVEL="&gt;partition:graph_fuser:&gt;&gt;kernel_cache" \
python your_script.py &amp;&gt; log</code></p>
<p>This logs all TorchScript IR parsed to codegen IR as well as kernel generated and executed by nvfuser. Since fallback path is disabled, it's likely that the last log would indicate the failing fusion.</p>
<p>Hint: look for last <code>Before Compilation:</code> that indicates a parsing failure, or <code>running GraphCache: xxxxx</code>, which indicates jit compilation/execution failure (also search for the GraphCache address, which would should have dumped a TorchScript IR earlier.</p>
</li>
</ol>
<h3>Query nvfuser codegen kernels</h3>
<p>There're a few debug dump that could be turned on via environment variables. Look for <code>PYTORCH_NVFUSER_DUMP</code> inside <code>[pytorch_source_path]/torch/csrc/jit/codegen/cuda/utils.cpp</code>. A few useful ones are:
1. <code>dump_eff_bandwidth</code>: print out effective bandwidth of each generated kernel. This naively measure the kernel time divided by I/O buffer size and is a good/simple metric of performance for bandwidth bound kernels
2. <code>cuda_kernel</code>: print out generated cuda kernels
3. <code>launch_param</code>: print out launch config of generated kernels
4. <code>kernel_args</code>: print out input/output/buffer tensors of all executed codegen kernels, note that for buffers, we indicate whether they are zero-initialized, which hints on an extra kernel to fill the tensor before codegen kernels.</p>
<h3>FAQs</h3>
<ol>
<li>There's regression after turning on nvfuser.</li>
</ol>
<p>First thing is to check that you have fusion kernel running properly. Try to run your model with fallback disabled to see if you hit any errors that caused fallback via <code>export PYTORCH_NVFUSER_DISABLE=fallback</code>.</p>
<p>If turning on NVFuser produces unexpected outputs, set the <code>PYTORCH_NVFUSER_DISABLE</code> environment variable to disable some of the optional features, e.g.:
- <code>fma</code>: disable using FMA instructions
- <code>index_hoist</code>: disable optimization to hoist common index expressions
- <code>predicate_elimination</code>: disable optimization to eliminate redundant predicates
- <code>unroll_with_rng</code>: disable unrolling when RNG is used</p>
<p>For example, <code>export PYTORCH_NVFUSER_DISABLE=fma,index_hoist</code> would disable FMA and index hoisting.</p>
<ol>
<li>I didn't see any speedup with nvfuser.</li>
</ol>
<p>Check if there is fusion in your script model. Run your script with <code>PYTORCH_JIT_LOG_LEVEL="graph_fuser"</code>, you should see some log dump of before/after graph regarding fusion pass. If nothing shows up in the log, that means something in TorchScript is not right and fusion pass are not executed. Check [General ideals of debug no-fusion] for more details.</p>
<ol>
<li>I ran into codegen issues with nvfuser, how do I disable nvfuser?</li>
</ol>
<p>There are three ways to disable nvfuser. Listed below with descending priorities:</p>
<ul>
<li>Force using NNC instead of nvfuser for GPU fusion with env variable <code>export PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1</code>.</li>
<li>Disabling nvfuser with torch API <code>torch._C._jit_set_nvfuser_enabled(False)</code>.</li>
<li>
<p>Disable nvfuser with env variable <code>export PYTORCH_JIT_ENABLE_NVFUSER=0</code>.</p>
</li>
<li>
<p>Is there any more knobs to tune nvfuser fusion?</p>
</li>
</ul>
<p>Some opt-out features in nvfuser are exposed via env var <code>PYTORCH_NVFUSER_DISABLE</code>. e.g. <code>fallback</code> to disable aten fallback during compilation failure and <code>fma</code> to disable fused multiply-add, you would set <code>export PYTORCH_NVFUSER_DISABLE="fallback,fma"</code>. Note that disabling fma would usually regress on performance so we strongly encourage to not disable it.</p>
<p>There's also opt-in features via env var <code>PYTORCH_NVFUSER_ENABLE</code>.
- <code>complex</code> would enable complex floating type support in nvfuser (currently experimental and turned off by default to avoid functional regression);
- <code>linear_decomposition</code> enables decomposition of the bias add in linear layer. Similarly, <code>conv_decomposition</code> enables decomposition of the bias add in conv layer. In some small benchmark models, we noticed that such decompositions added more overhead in compilation that out-weighs the benefit of faster kernel. Hence we decided to change these to be opt-in instead.</p>