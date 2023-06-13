<h1>Instruction count microbenchmarks</h1>
<h2>Quick start</h2>
<h3>To run the benchmark:</h3>
<p>```</p>
<h1>From pytorch root</h1>
<p>cd benchmarks/instruction_counts
python main.py
```</p>
<p>Currently <code>main.py</code> contains a very simple threadpool (so that run time isn't
unbearably onerous) and simply prints the results. These components will be
upgraded in subsequent PRs.</p>
<h3>To define a new benchmark:</h3>
<ul>
<li><code>TimerArgs</code>: Low level definition which maps directly to
<code>torch.utils.benchmark.Timer</code></li>
<li><code>GroupedStmts</code>: Benchmark a snippet. (Python, C++, or both) Can automatically
generate TorchScript and autograd variants.</li>
<li><code>GroupedModules</code>: Like <code>GroupedStmts</code>, but takes <code>nn.Module</code>s</li>
<li><code>GroupedVariants</code>: Benchmark-per-line to define many related benchmarks in a
single code block.</li>
</ul>
<h2>Architecture</h2>
<h3>Benchmark definition.</h3>
<p>One primary goal of this suite is to make it easy to define semantically
related clusters of benchmarks. The crux of this effort is the
<code>GroupedBenchmark</code> class, which is defined in <code>core/api.py</code>. It takes a
definition for a set of related benchmarks, and produces one or more concrete
cases. It's helpful to see an example to understand how the machinery works.
Consider the following benchmark:</p>
<p>```</p>
<h1><code>GroupedStmts</code> is an alias of <code>GroupedBenchmark.init_from_stmts</code></h1>
<p>benchmark = GroupedStmts(
    py_stmt=r"y = x * w",
    cpp_stmt=r"auto y = x * w;",</p>
<pre><code>setup=GroupedSetup(
    py_setup="""
        x = torch.ones((4, 4))
        w = torch.ones((4, 4), requires_grad=True)
    """,
    cpp_setup="""
        auto x = torch::ones((4, 4));
        auto w = torch::ones((4, 4));
        w.set_requires_grad(true);
    """,
),

signature="f(x, w) -&gt; y",
torchscript=True,
autograd=True,
</code></pre>
<p>),
```</p>
<p>It is trivial to generate Timers for the eager forward mode case (ignoring
<code>num_threads</code> for now):</p>
<p>```
Timer(
    stmt=benchmark.py_fwd_stmt,
    setup=benchmark.setup.py_setup,
)</p>
<p>Timer(
    stmt=benchmark.cpp_fwd_stmt,
    setup=benchmark.setup.cpp_setup,
    language="cpp",
)
```</p>
<p>Moreover, because <code>signature</code> is provided we know that creation of <code>x</code> and <code>w</code>
is part of setup, and the overall computation uses <code>x</code> and <code>w</code> to produce <code>y</code>.
As a result, we can derive TorchScript'd and AutoGrad variants as well. We can
deduce that a TorchScript model will take the form:</p>
<p><code>@torch.jit.script
def f(x, w):
    # Paste `benchmark.py_fwd_stmt` into the function body.
    y = x * w
    return y  # Set by `-&gt; y` in signature.</code></p>
<p>And because we will want to use this model in both Python and C++, we save it to
disk and load it as needed. At this point Timers for TorchScript become:</p>
<p>```
Timer(
    stmt="""
        y = jit_model(x, w)
    """,
    setup=""",
        # benchmark.setup.py_setup
        # jit_model = torch.jit.load(...)
        # Warm up jit_model
    """,
)</p>
<p>Timer(
    stmt="""
        std::vector<torch::jit::IValue> ivalue_inputs(
            torch::jit::IValue({x}),
            torch::jit::IValue({w})
        );
        auto y = jit_model.forward(ivalue_inputs);
    """,
    setup="""
        # benchmark.setup.cpp_setup
        # jit_model = torch::jit::load(...)
        # Warm up jit_model
    """,
)
```</p>
<p>While nothing above is particularly complex, there is non-trivial bookkeeping
(managing the model artifact, setting up IValues) which if done manually would
be rather bug-prone and hard to read.</p>
<p>The story is similar for autograd: because we know the output variable (<code>y</code>)
and we make sure to assign it when calling TorchScript models, testing AutoGrad
is as simple as appending <code>y.backward()</code> (or <code>y.backward();</code> in C++) to the
stmt of the forward only variant. Of course this requires that <code>signature</code> be
provided, as there is nothing special about the name <code>y</code>.</p>
<p>The logic for the manipulations above is split between <code>core/api.py</code> (for
generating <code>stmt</code> based on language, Eager/TorchScript, with or without AutoGrad)
and <code>core/expand.py</code> (for larger, more expansive generation). The benchmarks
themselves are defined in <code>definitions/standard.py</code>. The current set is chosen
to demonstrate the various model definition APIs, and will be expanded when the
benchmark runner infrastructure is better equipped to deal with a larger run.</p>
<h3>Benchmark execution.</h3>
<p>Once <code>expand.materialize</code> has flattened the abstract benchmark definitions into
<code>TimerArgs</code>, they can be sent to a worker (<code>worker/main.py</code>) subprocess to
execution. This worker has no concept of the larger benchmark suite; <code>TimerArgs</code>
is a one-to-one and direct mapping to the <code>torch.utils.benchmark.Timer</code> instance
that the worker instantiates.</p>