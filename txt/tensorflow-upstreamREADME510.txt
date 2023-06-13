<h1>TensorFlow Graph IR</h1>
<p>This directory contains the definition of the Intermediate Representation (IR)
for TensorFlow graphs using MLIR.</p>
<h2>Introduction</h2>
<p>This directory defined an MLIR dialect, the “TensorFlow Graph dialect”, that
represents accurately TensorFlow graphs. Contrary to the previous TensorFlow
dialect which made some opinionated choices that diverged from GraphDef and
TensorFlow Graph semantics, this dialect embraces TensorFlow Graph as it is. In
particular the concepts of control dependencies, requested device, assigned
device, and node name are all first-class attributes on the MLIR operations in
this dialect.</p>
<p>The main principle that drove the development of this dialect has been to ensure
perfect round-trip and general compatibility with existing TensorFlow semantics,
so that this solution can be deployed by default in any situation where "Graph
Optimization" and Grappler transformations are involved today, regardless of
TensorFlow V1 or V2. This new approach is also made possible by evolutions in
MLIR that allow representing graphs in a way that wasn’t possible before (more
in the <a href="#graph_operation_design">Graph operation design</a> section below).</p>
<h2>History of Dialects for TensorFlow</h2>
<p>MLIR started with a basic structure reflecting LLVM in that it defined a
<code>Module</code> containing a list of <code>Functions</code>. Each of these was defining a body
constrained to be a Control-Flow Graph (CFG): a list of <code>Blocks</code>, each of them
containing a list of <code>Operations</code>. A fundamental aspect of the CFG
representation is the notion of “control”: the abstract semantic model considers
that a single <code>Operation</code> executes at a given time, and the next <code>Operation</code> to
execute is necessarily the one listed immediately after[^1]. The last
<code>Operation</code> in a <code>Block</code> is a <code>Terminator</code>: it decides what is the next <code>Block</code>
where the control will be transferred (think of a branch).</p>
<p>When MLIR started, a first dialect -- that we were referring to as “TF control
dialect” -- was developed to model TensorFlow graphs. This dialect supported
control dependencies, but didn’t allow cycles in the graph, which forced some
tricks to model TensorFlow V1 loops and in particular the <code>NextIteration</code>
operation. While this dialect enabled some experimentation, it wasn’t seen as
really practical and another dialect was co-existing: the “tf” dialect that
we’re using currently. This dialect was designed before TF2.0
<a href="https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html">was released</a>,
and made strong assumptions about TensorFlow evolving towards a world where
eager execution and function execution become unified and V1 specific constructs
would be deprecated and disappear. As such control dependencies are not
supported and are instead implicit, control-flow V1 ops (such as Switch &amp; Merge)
and deadness aren’t supported[^2], new device placement modelling solutions were
considered. These choices in the model enabled us to write graph transformations
as stateless DAG-to-DAG patterns that can be applied to a subgraph, without
considering the entire graph.</p>
<h2>Motivation</h2>
<p>The combination of the TensorFlow and executor dialects allows for importing
most TensorFlow graphs and the TensorFlow dialect has proven enough to implement
the TF/XLA bridge, TFLite converter, and TFRT . However, the intent was for
TensorFlow 2.0 to trace TensorFlow functions directly in the TensorFlow dialect,
leaving the executor dialect only as a way to provide limited support for
TensorFlow V1 graphs.</p>
<p>However, the implementation of TensorFlow 2.0 didn't break away from TensorFlow
V1 entirely, instead TensorFlow functions are wrapped above TensorFlow V1 and
expose a leaky abstraction over the classical graph. As a result, the TensorFlow
dialect never got in a position to be enabled by default in TensorFlow. In
particular there are many subtle way in which TensorFlow functions diverges from
the sequential eager interpretation. For example the following pattern has been
recommended to users who intended to call a function <code>bar</code> knowing that the
first argument wasn’t necessary if they only used the first result.</p>
<p><code>@tf.function
  def foo(z):
    x = tf.Placeholder(tf.int32)
    y, _ = bar(x, z)
    return y</code></p>
<p>The use of a placeholder would throw an exception in eager mode, but “works” in
graph mode as long as inlining and pruning ensure the placeholder is removed
before execution.</p>
<p>Other cases involve the need for control dependencies beyond what the
auto-control-dependency tracking offers. For example the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/custom_gradient.py#L497">tf.recompute_grad</a>
creates control-dependencies on non-side-effecting ops to have a finer grain
control of memory usage.</p>
<p>Finally, the error modelling in TensorFlow can also be surprising. While in
eager op-by-op mode the execution is interrupted as soon as an error occurs,
<code>tf.function</code> tracing does not consider error handling as side-effecting
(otherwise it would have to add a control dependency between every node!) and as
such a program like:</p>
<p><code>@tf.function
def foo(x, y, variable):
   b = tf.matmul(x, y)
   variable.assign(1.0)
   return b</code></p>
<p>Does not guarantee that the assignment to the variable won’t occur if an error
occurs while processing the matmul, so calling:</p>
<p><code>foo(1., 2., variable)</code></p>
<p>Throws an exception because <code>tf.matmul</code> expects rank-2 tensors, but the variable
may or may not have been assigned. As such a user may want to opt in a safer
behavior for their function:</p>
<p><code>@tf.function
def foo(x, y, variable):
   b = tf.matmul(x, y)
   with tf.control_dependencies([b]):
     variable.assign(1.0)
   return b</code></p>
<p>However, this control dependency cannot be modelled in the TensorFlow dialect:
it will be just dropped! There is no solution today to prevent the variable
assignment to be executed ahead of the <code>matmul</code> in the TensorFlow Dialect.</p>
<p>While many of these cases could be modeled with different constructs at the
source level, this would be a major overhaul of TensorFlow itself, and more
importantly its ecosystem. Instead, we recognize that the TensorFlow dialect as
it exists today cannot support all of these use-cases, and it prevented MLIR
from providing a general graph transformation solution for TensorFlow,
contributing to more fragmentation instead of reducing it as promised.</p>
<p>The rest of this document describe how this new dialect follows a more pragmatic
approach to enable MLIR deployment in TensorFlow.</p>
<h2>Design</h2>
<p>This new dialect intends to allow us to replace Grappler and existing graph
transformations, for TensorFlow V1 and V2 without constraints. As such the main
principle is to support perfect roundtrip between TensorFlow Graph/GraphDef and
MLIR.</p>
<h3>General Operations</h3>
<p>An individual TensorFlow <code>NodeDef</code> is translated into an individual MLIR
operation using the following form:</p>
<p><code>%AddV2, %ctl = tfg.AddV2(%placeholder, %placeholder_1) [%ctl_1, %ctl_2]
                     device("GPU") assigned_device("TPU") name("add")
                     {some_attribute = "some attr!"}
                     : (tensor&lt;*xi32&gt;, tensor&lt;*xi32&gt;) -&gt; (tensor&lt;*xi32&gt;)</code></p>
<ul>
<li>Each operation returns an optional variadic number of tensors as well as a
    control token to express potential control dependencies.</li>
<li>The node type is carried in the operation mnemonic.</li>
<li>The list of regular inputs is in-between parentheses.</li>
<li>Optional control dependencies are exposed after the regular inputs and
    printed between square brackets.</li>
<li>The pre-placement “requested device” as well as the post-placement “assigned
    device” information are preserved.</li>
<li>The node name is carried as a first-class attribute.</li>
<li>Optional “op specific” attributes can be listed between curly brackets.</li>
<li>Finally, the type signature follows, omitting the control dependencies.</li>
</ul>
<p>This structure allows for a perfect round-trip to NodeDef, while still being
ergonomic when manipulating it in MLIR (compared to the <code>tf\_executor</code> dialect
for example). The tradeoff we are making here is that we preserve all
attributes, including the “derived” ones[^3], which creates some amount of
redundancy with the signature. We may consider pruning these redundant
attributes in the future in the same way as we do in the TensorFlow dialect.</p>
<h3>Graph Operation</h3>
<p>A structural operation is introduced as a container: <code>tfg.graph</code> acts as a bag
of unordered TensorFlow operations, and carries a “version” attribute that
corresponds to the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/versions.proto">VersionDef</a>
present in GraphDef:</p>
<p><code>tfg.graph #tfg.version&lt;producer = 42, min_consumer = 33&gt; {
  %arg0, %ctl_0 = tfg.placeholder() : () -&gt; (tensor&lt;*xi32&gt;)
  %add, %ctl_1 = tfg.AddV2(%arg0, %arg1)
                    : (tensor&lt;*xi32&gt;, tensor&lt;*xi32&gt;) -&gt; (tensor&lt;*xi32&gt;)
  %arg1, %ctl_2 = tfg.placeholder() : () -&gt; (tensor&lt;*xi32&gt;)
}</code></p>
<p>Note that the <code>AddV2</code> operation is using the result of a <code>placeholder</code> operation
that is defined later in the list. This wasn’t possible in MLIR 2 years ago when
the TensorFlow dialect was designed. It was actually
<a href="https://groups.google.com/a/tensorflow.org/g/mlir/c/gPQFIy9XpVw/m/hfxmBGF8AQAJ">attempted to allow such unordered semantics</a>
and break away from the CFG-centric representation, but we couldn’t reach a
consensus, and some key members of the team believed that a departure from
CFG/SSA would limit the reusability of many algorithms. On the other hand, this
choice prevented us to design a graph dialect that can just replace TensorFlow
Graph structure as-is. Since then MLIR evolved to become more general and this
feature is now available (it was motivated by the
<a href="https://llvm.discourse.group/t/rfc-allowing-dialects-to-relax-the-ssa-dominance-condition/833">support for HW synthesis tools</a>).
Another recent development that made it also more friendly is the
<a href="https://llvm.discourse.group/t/rfc-making-terminator-optional-for-single-block-graph-regions/2997">removal of the requirement for terminators</a>:
the <code>tfg.graph</code> operation above contains a single block listing operations, and
a terminator does not have any role to play. Finally, a Dialect can now
<a href="https://llvm.discourse.group/t/rfc-dialect-fallback-for-opinterface/3074">act as fallback for OpInterfaces</a>,
which allows us to reuse more of the TensorFlow registry to provide information
to MLIR passes about TensorFlow operation without having to register them with
MLIR in the first place.</p>
<p>The <code>tfg.graph</code> operation round-trips almost perfectly to
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/graph/graph.h#L504">Graph</a>,
except for the <code>Function Library</code>, which I address below.</p>
<h3>Function Library</h3>
<p>Functions in TensorFlow are stored as
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/function.proto">FunctionDef</a>,
which has a signature, holds attributes, identifies argument and returned
values, and finally contains a list of nodes for its body. While on the surface
this <code>repeated NodeDef node_def</code> field looks identical to the body of
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto#L17">GraphDef</a>,
there are fundamental differences in the representation, and in particular the
format the edges are represented is different.</p>
<p>To understand these differences, it is important to realize that a key aspect of
<code>FunctionsDef</code> is that they are stored uninstantiated, and can be considered in
a similar way to a C++ template function. The signature is actually an <code>OpDef</code>,
and just like any regular TensorFlow operation the types of the arguments and
the results are encoded and constrained with attributes. These attributes are
only provided or inferred based on the function’s use: the call-site is
responsible for instantiating a function before it’s body can be represented as
a Graph. Because of this, the body of an uninstantiated function is modeled
differently than Graph body:</p>
<p><code>tfg.func generic @foo(%arg0 : !tfg.tensor {tfg.name = "input"},
                        %arg1 : !tfg.tensor {tfg.name = "another_input"})
      -&gt; (!tfg.tensor {tfg.name = "result1"},
          !tfg.tensor {tfg.name = "result2"})
      attributes {description = "function foo"} {
    %Greater, %ctl_0 = tfg.Greater(%arg0, %arg1) name("Greater")
    %G_z = tfg.get_result(%Greater) "z" : 0
    %Switch, %ctl_1 = tfg.Switch(%G_z, %G_z) name("cond/Switch")
    %s_true = tfg.get_result %Switch "output_true" : 0
    %s_false = tfg.get_result %Switch "output_false" : 0
    tfg.return(%s_true, %s_false) [%ctl_0]
  }</code></p>
<p>Note how the tensor types <code>!tfg.tensor</code> are opaque, and every operation returns
a single tensor output and a control token. The tensor output is then unpacked
by looking up individual results by name. This is particularly visible with the
<code>Switch</code> operation where the two results are accessed using <code>tfg.get_result</code>
looking them up by name <code>output_true:0</code> and <code>output_false:0</code>. This is required
because the OpDef can define the number of output based on the attribute present
on the NodeDef, and these attributes can in turn be dependent on the attributes
added on the function during instantiation (you can read more about it in the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto#L48-L55">description of the placeholder attribute value</a>).</p>
<p>Post-instantiation, a function body is similar to the one of a graph:</p>
<p><code>tfg.func @foo(%arg0 : tensor&lt;*xf32&gt; {tfg.name = "input"},
                %arg1 : tensor&lt;*xf32&gt; {tfg.name = "another_input"})
      -&gt; (tensor&lt;*xi1&gt; {tfg.name = "result1"},
          tensor&lt;*xi1&gt; {tfg.name = "result2"})
      attributes {description = "function foo"} {
    %Greater, %ctl_0 = tfg.Greater(%arg0, %arg1) [%arg1.ctl] name("Greater")
                          : (tensor&lt;*xf32&gt;, tensor&lt;*xf32&gt;) -&gt; tensor&lt;*xi1&gt;
    %Switch:2, %ctl_1 = tfg.Switch(%Greater, %Greater) name("cond/Switch")
                          : (tensor&lt;*xi1&gt;, tensor&lt;*xi1&gt;) -&gt; tensor&lt;*xi1&gt;
   tfg.return(%Switch#0, %Switch#1) [%ctl_0]
  }</code></p>
<p>The operations aren’t ordered, except for the <code>tfg.return</code> which is a terminator
and must be the last operation. The only remaining difference with a graph is in
the handling of the function signature (arguments and returned values), and
attributes.</p>
<p>There is one aspect of the modelling worth mentioning from the MLIR point of
view: FunctionDef allows for nodes in a graph to express input control
dependencies from function arguments. However, in MLIR you need an actual
<a href="https://en.wikipedia.org/wiki/Static_single_assignment_form">SSA</a> value to add
an edge between two operations. These values are typed and this is why
operations define a control token (like <code>%ctl_0</code>). We apply the same recipe for
arguments and for each of them we define a control token. We omit these “shadow
arguments” from the textual form, but in-memory the MLIR function has really 4
arguments:</p>
<p><code>tfg.func @foo(%arg0 : tensor&lt;*xf32&gt; {tfg.name = "input"}, %arg0.ctl : !tfg.control
      %arg1 : tensor&lt;*xf32&gt; {tfg.name = "another_input"}, %arg1.ctl : !tfg.control)
      -&gt; (tensor&lt;*xi1&gt; {tfg.name = "result1"},
          tensor&lt;*xi1&gt; {tfg.name = "result2"})
      attributes {description = "function foo"} {
   ...</code></p>
<p>The convention is that callers are only exposed to the non-control input
(<code>%arg0</code> and <code>%arg1</code>) while the control tokens are only intended to be visible
and used in the body. This makes it very aligned with how TensorFlow works.
Inside the body, values for the control dependencies on the arguments are
available with a <code>.ctl</code> suffix (i.e. <code>%arg0.ctl</code> and <code>%arg1.ctl</code>).</p>
<h3>Saved Model</h3>
<p>The basic blocks above are enough to model <code>GraphDef</code>, but not the entirety of
SavedModel. However, most of the use cases that we’re targeting right now are in
the scope of the existing GraphOptimization and Grappler APIs, which aren’t
really coupled to SavedModel. The user can load a SavedModel independently of
MLIR and invoke MLIR transformations on a Function or Graph from there. There is
also already a dialect to model the specific aspects of SavedModel, it is
currently wrapping around the TensorFlow executor dialect and the TensorFlow
dialect, and we may look into integrating it with the <code>tfg</code> dialect in the
future. For these reasons, we mostly leave out modeling the Saved Model for
future work right now.</p>
<h3>Future Enhancements</h3>
<p>Functional control-flow is modeled with nodes in the graph invoking functions in
the library. MLIR supports <code>region</code>s, which is a concept that allows attaching
subgraphs directly inside a graph, making it more friendly to optimizations. For
example a conditional operation can represent the two branches subgraph in the
TensorFlow dialect directly as follows:</p>
<p><code>%0, %1, %2 = "tf.IfRegion"(%arg0) ({
     %t0 = "tf.Abs"(%arg1) : (tensor&lt;2xf32&gt;) -&gt; tensor&lt;2xf32&gt;
     %t1 = "tf.Acos"(%arg1) : (tensor&lt;2xf32&gt;) -&gt; tensor&lt;2xf32&gt;
     %t2 = "tf.Acosh"(%arg1) : (tensor&lt;2xf32&gt;) -&gt; tensor&lt;2xf32&gt;
    "tf.Yield"(%t0, %t1, %t2) : (tensor&lt;2xf32&gt;, tensor&lt;2xf32&gt;, tensor&lt;2xf32&gt;) -&gt; ()
  }, {
     %e0 = "tf.Neg"(%arg1) : (tensor&lt;2xf32&gt;) -&gt; tensor&lt;2xf32&gt;
     %e1 = "tf.Relu"(%arg1) : (tensor&lt;2xf32&gt;) -&gt; tensor&lt;2xf32&gt;
     %e2 = "tf.Sin"(%arg1) : (tensor&lt;2xf32&gt;) -&gt; tensor&lt;2xf32&gt;
     "tf.Yield"(%e0, %e1, %e2) : (tensor&lt;2xf32&gt;, tensor&lt;2xf32&gt;, tensor&lt;2xf32&gt;)
  }): (tensor&lt;i1&gt;) -&gt; (tensor&lt;2xf32&gt;, tensor&lt;2xf32&gt;, tensor&lt;2xf32&gt;)
  %3 = "tf.Add"(%0, %1) : (tensor&lt;2xf32&gt;, tensor&lt;2xf32&gt;) -&gt; tensor&lt;2xf32&gt;
  %4 = "tf.Add"(%2, %3) : (tensor&lt;2xf32&gt;, tensor&lt;2xf32&gt;) -&gt; tensor&lt;2xf32&gt;</code></p>
<h2>Integration</h2>
<p>MLIR transformations in this dialect will operate on a module that will contain
at most one <code>graph</code> operation as well as a list of functions. This interface
will make such transformations suitable for fit within Grappler or as
GraphOptimization interchangeably.</p>
<p>Instead of a flat graph, an entry function will be provided when feeds/fetches
are available for the main graph (PRE_PLACEMENT graph optimizations execute in
Session before feeds/fetches are provided).</p>
<h2>FAQ</h2>
<h3>Why not just use the TensorFlow Executor Dialect?</h3>
<p>The executor dialect wasn’t designed to write transformation: it is designed as
a wrapper around the TensorFlow dialect: the intent was for it to be a stepping
stone to integrate MLIR and TensorFlow, and then disappear when TensorFlow V1
graphs would be deprecated. This new dialect embraces TensorFlow as it is
instead of as I wish it would be.</p>
<p>In particular the executor dialect represents each TensorFlow node as an
isolated “subgraph” nested under an “island” operation. This requires 3
operations and an extra region for each TensorFlow node, which is quite
inefficient in memory as well as requiring extra indirection when pattern
matching or updating nodes in the graph.</p>
<h3>What happens to the existing TensorFlow Dialects?</h3>
<p>The existing TensorFlow dialect is suitable for representing a large subset of
TensorFlow programs (like models that intend to convert to TFLite, or XLA), and
for such cases we will continue to use it.</p>
<h3>What happens to the existing TensorFlow Executor Dialect?</h3>
<p>This new TensorFlow Graph Dialect could be used to replace the Executor Dialect
as the standalone staging importing format. Importing from GraphDef/Graph would
always go through the TensorFlow Graph Dialect before using some clustering or
promotion algorithms to raise some subgraphs to the TensorFlow Dialect, just
like we do now to cluster islands operations in TensorFlow Executor Dialect.
The details of such mechanisms are left for future work.</p>
<!-- Footnotes -->

<p>[^1]: While the semantic model is sequential, this does not prevent an
    implementation to execute operation in parallel when proven safe. This is
    similar to how a superscalar CPU involves implicit parallelism. For
    example when mapping the TensorFlow dialect to TFRT, only side-effecting
    operations (Variable accesses for example) are sequenced.
[^2]: One of the first tools built with this was the TF-&gt;TFlite converter
    (replacing TOCO). Since V1 control-flow isn’t supported on TFLite this
    wasn’t a limitation.
[^3]: Derived attributes is a concept used in the TensorFlow dialect: since MLIR
    models type and shape information on each individual result produced by an
    operation, some attributes that are inserted for the sole purpose of
    typing are redundant and eliminated in MLIR.</p>