<blockquote>
<p>:warning: <strong>This is an experimental feature</strong></p>
</blockquote>
<h1>Static Runtime</h1>
<p>Static Runtime is an optimized CPU inference runtime for PyTorch models.
It can be used as a drop-in replacement for the TorchScript JIT interpreter
in either C++ or Python.</p>
<p>Static Runtime is mainly useful if the following conditions are met:
1. The model has very little control flow.
2. PyTorch overhead (tensor creation, etc) accounts for
a non-trivial fraction of the model's runtime. In particular, if
tensor allocation consumes a significant amount of time, Static
Runtime can help. Memory for intermediate tensors is coalesced into
a single slab, so most dynamic allocations are avoided during
inference.
3. Inference performance is extremely important.</p>
<h2>Assumptions</h2>
<p>This is a list of current assumptions for use with
this feature.</p>
<ul>
<li>Inference only execution, CPU only</li>
<li>Static input dtypes</li>
<li>Static input shapes (the runtime supports dynamic shapes, but excessive dynamic shapes may degrade performance)</li>
</ul>
<h2>Threading model</h2>
<p>Static runtime supports two execution modes.</p>
<p>Mode 1: single-threaded with no parallelism except for intra-op parallelism.
For this mode, you can do either:
<code>// m is the TorchScript module
  auto runtime = StaticRuntime(m, opts);
  auto output = runtime.run(args, kwargs);</code>
or
<code>auto mod = PrepareForStaticRuntime(m);
  auto runtime = StaticRuntime(mod, opts);
  auto output = runtime.run(args, kwargs);</code>
Mode 2: similar to data parallelism, run the same model for different inputs
on different threads at the same time. In this case, run
<code>PrepareForStaticRuntime</code> to prepare the graph for Static Runtime. You
should have one InferenceModule instance per model, and one Static Runtime instance
per running thread. To avoiding creating StaticRuntime on the fly, use a
synchronized stack (i.e. <code>boost::lockfree::stack</code>) to cache all the Static
Runtime instances in your code.
```
  // initialization
  auto mod = PrepareForStaticRuntime(m);
  // 128 is good for most cases. Pick a number that works for you
  boost::lockfree::stack<std::shared_ptr\<StaticRuntime>,
    boost::lockfree::fixed_sized\<true>> pool(128);</p>
<p>// inference
  std::shared_ptr<StaticRuntime> runtime = nullptr;
  pool.pop(runtime);
  if (!runtime) {
    runtime = std::make_shared<StaticRuntime>(mod, opts);
  }
  auto output = runtime-&gt;run(args, kwargs);
  pool.push(runtime);
```</p>
<p><strong>In both modes, <code>StaticRuntime</code> may not be used after its associated <code>StaticModule</code> is destructed!</strong></p>
<h2>Memory Planning</h2>
<p>Static runtime's memory planner does two things:</p>
<p>1) Coalesces internal allocations for tensor storage
2) Does static analysis to figure out how to efficiently re-use memory.</p>
<h3>Standard Resizing</h3>
<p>Static runtime will record the space required for each intermediate managed tensor it sees
on the first inference iteration. An intermediate tensor is <em>managed</em> if two conditions
are satisfied:</p>
<p>1) The op that produces it has an out variant. Out variants are wrappers around ops that
conceptually transform the op's signature from <code>Tensor some_op(const Tensor&amp; some_arg)</code>
into <code>void some_op(Tensor&amp; output, const Tensor&amp; some_arg)</code>. Out variants are registered
with static runtime via the <code>REGISTER_OPERATOR_FUNCTOR</code> macro; see "Registering Ops" for
more info.</p>
<p>2) The tensor does not alias a graph output. Output tensors are handled separately by
the memory planner, see "Managed Output Tensors" for details.</p>
<p>With this algorithm, static analysis is used to group the tensors in <code>StorageGroup</code>s.
Tensors in the same storage group share memory, and two tensors can be in the same storage group
if their lifetimes do not overlap.</p>
<p>On the subsequent iterations, static runtime allocates the tensor buffer at the start of the run.
The amount of memory allocated is <code>sum([max(tensor.size()) for tensor in storage_groups])</code>.</p>
<p>If a tensor needs to be bigger than the allocated space on subsequent runs, a dynamic allocation
will occur. This is why dynamic shapes will degrade performance. With the standard resizing
strategy, static runtime will record the new largest tensor size in each storage group at the
end of the iteration and allocate a buffer that is possibly bigger on the next iteration.</p>
<h3>Managed Output Tensors</h3>
<p><code>StaticRuntime</code> can optionally manage output tensors via the <code>manage_output_tensors</code> option in <code>StaticModuleOptions</code>.
When this flag is turned on, we coalesce allocations for output tensors together. Note that the buffer containing
output tensors is separated from the one containing intermediate tensors. The former needs to live past the end
of the inference run, but the latter needs deallocated at the end of the run.</p>
<p>Under the hood, we store a refcounted pointer to the output arena in each returned <code>Tensor</code>. The arena is destroyed
explicitly.</p>
<h2>Registering Ops</h2>
<p>Static runtime has three op execution modes:</p>
<p>1) Out variants: ops that return tensors which we may be able to manage. See "Memory Planning" for more
details. Out variants are registered via the <code>REGISTER_OPERATOR_FUNCTOR</code> macro in <code>ops.h</code>.
<code>REGISTER_OPERATOR_FUNCTOR(
  aten::op_name,
  aten_op_name, // This macro generates a struct, this field names it
  [](torch::jit::Node* n) -&gt; SROperator {
    // This mechanism lets us support a subset of schemas
    if (n-&gt;matches(some_schema)) {
      return some_overload;
    } else if (n-&gt;matches(another_schema)) {
      return another_overload;
    }
    return nullptr;
  })</code></p>
<p>A <code>SROperator</code> is a type alias for <code>std::function&lt;void(ProcessedNode*)&gt;</code>. See "Implementation Details" for more
details on <code>ProcessedNode</code>.</p>
<p>2) Native functions: just like out variants, except their outputs cannot be managed. This is because the op's return
type is not a tensor or it is a view op (returns a tensor alias instead of a new tensor). Registration is done with
<code>REGISTER_NATIVE_OPERATOR_FUNCTOR</code>. This macro is used in the same way as <code>REGISTER_OPERATOR_FUNCTOR</code>.</p>
<p>3) JIT fallback: static runtime has no implementation for this op, so the implementation that the JIT interpreter uses
is selected instead.</p>
<p>When loading a model, ops are selected for each <code>torch::jit::Node</code> in the graph as follows:</p>
<p>1) If an out variant is registered, pass the node to the function that produces the <code>SROperator</code>. If
the result is not <code>nullptr</code>, use that op.
2) If a native function is registered, pass the node to the function that produces the <code>SROperator</code>. If
the result is not <code>nullptr</code>, use that op.
3) Use the JIT implementation. Static runtime will throw an exception if it does not exist.</p>
<h2>Implementation Details</h2>
<h3>Structure and Lifetime Details</h3>
<p>The following diagram shows the core data structure. An arrow from <code>A</code> to <code>B</code> means that
<code>A</code> stores a reference to <code>B</code>. If the reference is unowned,
<code>A</code> may not out live <code>B</code> or anything that <code>B</code> stores a reference to (directly or indirectly).
If the reference is owned, the lifetimes of <code>A</code> and <code>B</code> are the same.
```</p>
<pre><code>                     IValue array◄────────────────┐─────────────────────────────────────────┐
                          ▲                       │               Owns                      │       Owns
                          │                       │  ┌───────────────────────────────►ProcessedNode───────►BlockRunner
                          │Owns                   │  │                                      │                  │
                          │         Owns          │  │   Owns                               │                  │
</code></pre>
<p>StaticModule◄───────────StaticRuntime───────────►BlockRunner────────►MemoryPlanner              │                  ▼
    │     │                                           │                  │                      │                 ...
Owns│     │                                           │                  │                      │
    ▼     │                                           │                  │                      │
BlockInfo◄├───────────────────────────────────────────┘──────────────────┘                      │
          │                                                                                     │
      Owns│                                                                                     │
          ▼                                                                                     │
ProcessedFunction ◄─────────────────────────────────────────────────────────────────────────────┘
```</p>
<p>Each class is described in detail below.</p>
<h3><code>StaticModule</code> and <code>StaticRuntime</code></h3>
<p><code>StaticModule</code>s are constructed from <code>torch::jit::Module</code>s and can be used to construct <code>StaticRuntime</code>
instances. Each <code>StaticModule</code> caches exactly one <code>StaticRuntime</code> instance - it is lazily initialized when
you access it via <code>runtime()</code>.</p>
<p><code>StaticModule::operator()</code> can be used directly to make predictions. Under the hood, this method just
forwards to the cached runtime's <code>StaticRuntime::operator()</code>. One upshot of this behavior is that
<code>StaticModule::operator()</code> is not thread-safe.</p>
<p>The way to use static runtime in a multi-threaded context is to give each thread its own <code>StaticRuntime</code>
instance. New runtime instances can be created directly (<code>StaticRuntime(static_module)</code>) or <code>clone()</code>'d from
an existing runtimes.</p>
<p><code>StaticModule</code> takes a set of options that control the behavior of the runtime instances that it spawns;
see <code>StaticModuleOptions</code> for more details.</p>
<p>Internally, <code>StaticRuntime</code> owns an array of <code>IValue</code>s that is referenced from all <code>BlockRunner</code>s and
<code>ProcessedNode</code>s. All values that are generated at runtime are stored in this array.</p>
<h3><code>BlockRunner</code></h3>
<p>A <code>BlockRunner</code> represents a single sub-block in the graph. Every graph has at least one <code>BlockRunner</code>
corresponding to the top-level block, and <code>StaticRuntime</code> starts its inference run by invoking
<code>(*top_level_block)(args, kwargs)</code>. Each <code>BlockRunner</code> has its own <code>MemoryPlanner</code> and set of <code>ProcessedNode</code>s.
Special nodes that have sub-blocks (like <code>prim::If</code>) might own <code>BlockRunner</code>s. The op implementations are responsible
for invoking <code>BlockRunner</code>s corresponding to sub-blocks.</p>
<h3><code>MemoryPlanner</code></h3>
<p>See the "Memory Planning" section. <code>MemoryPlanner</code> is an abstract base class. Each sub-class implements a different
memory planning algorithm.</p>
<p>In addition to the memory planning we do for tensors, <code>MemoryPlanner</code> encapsulates a few other optimizations.</p>
<ul>
<li>Managed output tensors (see "Managed Output Tensors")</li>
<li>Borrowed <code>IValue</code>s; ops that just unpack their inputs (e.g. <code>dict_unpack</code>) might produce weak-references to
avoid refcount bumps, the <code>MemoryPlanner</code> needs to destroy these borrows appropriately.</li>
</ul>
<h3><code>ProcessedNode</code> and <code>ProcessedFunction</code></h3>
<p><code>ProcessedNode</code> is our abstraction for a single op. Each <code>ProcessedNode</code> stores an unowned reference to <code>StaticRuntime</code>'s
<code>IValue</code> array. It knows how to map input/output indices to indices in this array (so <code>processed_node-&gt;output(i)</code> returns
a reference to <code>ivalue_array[some_set_of_indices[i]]</code>)</p>
<p>Each <code>ProcessedNode</code> stores a <code>ProcessedFunction</code>, which represents the actual op to execute. <code>ProcessedFunction</code>s are initialized
upon <code>StaticModule</code> construction according to the out variant/native/JIT fallback lookup rules described in "Registering Ops".
<strong>Note that all <code>ProcessedFunction</code>s are shared amongst all runtime instances</strong>, so all <code>ProcessedFunction</code>s must be thread-safe.</p>
<h3><code>ProcessedNodeMetadata</code></h3>
<p><code>ProcessedNodeMetadata</code> holds various "extra" fields on behalf of <code>ProcessedNode</code>. Typically, this field is unused. But a few ops need extra machinery to work:
* <code>prim::If</code> operations have two <code>BlockRunner</code>s for the execution of true and false sub-blocks depending upon the condition check.
* <code>prim::Loop</code> operations have a <code>BlockRunner</code> for the execution of the looping sub-block.
* <code>prim::fork</code> operations have <code>torch::jit::TaskLauncher</code> (<code>std::function&lt;void(std::function&lt;void()&gt;)&gt;</code>) responsible for forked graph execution.</p>
<h3>Asynchronous Execution</h3>
<p>The <code>StaticRuntime::runAsync()</code> API allows the execution of asynchronous operations on the <code>TaskLauncher</code> passed as arguments.
<code>StaticRuntime::runAsync()</code> performs inline execution of the parent graph on the caller thread. Asynchronous operations like <code>prim::fork</code> are executed
on the launcher passed in. In the case that no launcher is provided, the execution happens via <code>at::launch</code>, i.e. on the inter-op thread pool.</p>