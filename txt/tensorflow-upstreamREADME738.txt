<h1>XLA Interpreter Backend</h1>
<p>The XLA Interpreter backend operates at HLO-level by ingesting a HloModule and
evaluating the result of the HLO graph directly with HloEvaluator, without
lowering it further (to LLVM IR for example) before execution as other backends
(CPU and GPU for example) do.</p>
<p>Its key components are:</p>
<ul>
<li>[<code>InterpreterCompiler</code>] despite the inherited naming of "compiler", all
    <code>InterpreterCompiler</code> really does is the following:<ol>
<li>Runs certain HLO optimization passes on the given HLO graph.</li>
<li>Generates an <code>InterpreterExecutable</code> from the optimized HLO graph.</li>
<li>Registers itself in the global compiler factory registry.</li>
</ol>
</li>
<li>[<code>InterpreterExecutable</code>]: responsible for running input HLO graph through
    the <code>HloEvaluator</code>, allocating output buffer and finally copying evaluated
    Literal result over.</li>
<li>[<code>HloEvaluator</code>]: traverses a HLO graph and evaluates each node in DFS
    ordering along the way.</li>
</ul>