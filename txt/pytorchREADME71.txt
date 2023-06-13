<h1>FX Technical Overview</h1>
<p>FX is a toolkit for pass writers to facilitate Python-to-Python transformation of <code>nn.Module</code> instances. This toolkit aims to support a subset of Python language semantics—rather than the whole Python language—to facilitate ease of implementation of transforms. Currently, this feature is under a Beta release and its API may change.</p>
<h2>Table of Contents</h2>
<!-- toc -->

<ul>
<li><a href="#introduction">Introduction</a></li>
<li><a href="#use-cases">Use Cases</a></li>
<li><a href="#technical-details">Technical Details</a></li>
<li><a href="#internal-structure">Internal Structure</a></li>
<li><a href="#graph">Graph</a></li>
<li><a href="#node">Node</a></li>
<li><a href="#graphmodule">GraphModule</a></li>
<li><a href="#tracing">Tracing</a></li>
<li><a href="#symbolic-tracer">Symbolic Tracer</a></li>
<li><a href="#proxy">Proxy</a></li>
<li><a href="#torchdynamo">TorchDynamo</a></li>
<li><a href="#the-fx-ir-container">The FX IR Container</a></li>
<li><a href="#transformation-and-codegen">Transformation and Codegen</a></li>
<li><a href="#next-steps">Next steps</a></li>
</ul>
<!-- tocstop -->

<h1>Introduction</h1>
<h2>Use Cases</h2>
<p>FX should be used by pass writers to provide functionality for capturing and constructing nn.Module code in a structured way. We do not expect end users to utilize FX directly. A useful property of framing FX in this way is that passes can be seen as functions of the form <code>pass(in_mod : nn.Module) -&gt; nn.Module</code>. This means we can create composable pipelines of transformations.</p>
<p><img alt="An image of a sample nn.Module transformation pipeline that starts with a Quantize transformation, which is then composed with a Split transformation, then a Lower to Accelerator transformation" src="https://i.imgur.com/TzFIYMi.png" title="nn.Module transformation pipeline" /></p>
<p>In this example pipeline, we have a Quantize transformation, which is then composed with a Split transformation, then a Lower to Accelerator transformation. Finally, the transformed Modules are compiled with TorchScript for deployment. This last point emphasizes that not only should FX transforms be composable with each other, but their products are composable with other systems like TorchScript compilation or tracing.</p>
<p>By using <code>nn.Module</code> as the interface between passes, FX transforms are interoperable with each other, and the resulting model can be used anywhere an <code>nn.Module</code> can be used.</p>
<h2>Technical Details</h2>
<p>The following sections will walk us through the components that transform from original <code>torch.nn.Module</code> to FX IR and finally to generated Python code and a GraphModule instance:</p>
<p>FX’s front-end makes use of the dynamic nature of Python to intercept call-sites for various entities (PyTorch operators, Module invocations, and Tensor method invocations). The simplest way to get an FX graph is by using <code>torch.fx.symbolic_trace</code>.  We can see how this works by way of an example:</p>
<p>```python
import torch</p>
<p>class MyModule(torch.nn.Module):
  def <strong>init</strong>(self):
    super().<strong>init</strong>()
    self.param = torch.nn.Parameter(
        torch.rand(3, 4))
    self.linear = torch.nn.Linear(4, 5)</p>
<p>def forward(self, x):
    return self.linear(x + self.param).clamp(min=0.0, max=1.0)</p>
<p>from torch.fx import symbolic_trace
module = MyModule()
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)</p>
<p>input = torch.rand(3, 4)
torch.testing.assert_close(symbolic_traced(input), module(input))
```</p>
<p>Here, we set up a simple Module that exercises different language features: fetching a parameter, applying an arithmetic operator, applying a submodule (linear), and applying a Tensor method. <code>symbolic_trace</code> returns an instance of GraphModule, which is in itself a subclass of <code>nn.Module</code>. We can see that the <code>symbolic_traced</code> instance runs and returns the same result as the original module instance module.</p>
<h1>Internal Structure</h1>
<h2><a href="https://pytorch.org/docs/master/fx.html#torch.fx.Graph">Graph</a></h2>
<p>The <code>fx.Graph</code> is a core data structure in FX that represents the operations and their dependencies in a structured format. It consists of a List of <code>fx.Node</code> representing individual operations and their inputs and outputs. The Graph enables simple manipulation and analysis of the model structure, which is essential for implementing various transformations and optimizations.</p>
<h2>Node</h2>
<p>An <code>fx.Node</code> is a datastructure that represent individual operations within an <code>fx.Graph</code>, it maps to callsites such as operators, methods and modules. Each <code>fx.Node</code> keeps track of its inputs, the previous and next nodes, the stacktrace so you can map back the node to a line of code in your python file and some optional metadata stored in a <code>meta</code> dict.</p>
<h2><a href="https://pytorch.org/docs/master/fx.html#torch.fx.GraphModule">GraphModule</a></h2>
<p>The <code>fx.GraphModule</code> is a subclass of <code>nn.Module</code> that holds the transformed Graph, the original module's parameter attributes and its source code. It serves as the primary output of FX transformations and can be used like any other <code>nn.Module</code>. <code>fx.GraphModule</code> allows for the execution of the transformed model, as it generates a valid forward method based on the Graph's structure.</p>
<h1>Tracing</h1>
<h2><a href="https://pytorch.org/docs/master/fx.html#torch.fx.Tracer">Symbolic Tracer</a></h2>
<p><code>Tracer</code> is the class that implements the symbolic tracing functionality of <code>torch.fx.symbolic_trace</code>. A call to <code>symbolic_trace(m)</code> is equivalent to <code>Tracer().trace(m)</code>. Tracer can be subclassed to override various behaviors of the tracing process. The different behaviors that can be overridden are described in the docstrings of the methods on the class.</p>
<p>In the default implementation of <code>Tracer().trace</code>, the tracer first creates Proxy objects for all arguments in the <code>forward</code> function. (This happens in the call to <code>create_args_for_root</code>.) Next, the <code>forward</code> function is called with the new Proxy arguments. As the Proxies flow through the program, they record all the operations (<code>torch</code> function calls, method calls, and operators) that they touch into the growing FX Graph as Nodes.</p>
<h2>Proxy</h2>
<p>Proxy objects are Node wrappers used by the Tracer to record operations seen during symbolic tracing. The mechanism through which Proxy objects record computation is <a href="https://pytorch.org/docs/stable/notes/extending.html#extending-torch"><code>__torch_function__</code></a>. If any custom Python type defines a method named <code>__torch_function__</code>, PyTorch will invoke that <code>__torch_function__</code> implementation when an instance of that custom type is passed to a function in the <code>torch</code> namespace. In FX, when operations on Proxy are dispatched to the <code>__torch_function__</code> handler, the <code>__torch_function__</code> handler records the operation in the Graph as a Node. The Node that was recorded in the Graph is then itself wrapped in a Proxy, facilitating further application of ops on that value.</p>
<p>Consider the following example:</p>
<p>```python
  class M(torch.nn.Module):
      def forward(self, x):
          return torch.relu(x)</p>
<p>m = M()
  traced = symbolic_trace(m)
```</p>
<p>During the call to <code>symbolic_trace</code>, the parameter <code>x</code> is transformed into a Proxy object and the corresponding Node (a Node with op = “placeholder” and target = “x”) is added to the Graph. Then, the Module is run with Proxies as inputs, and recording happens via the <code>__torch_function__</code> dispatch path.</p>
<p>If you're doing graph transforms, you can wrap your own Proxy method around a raw Node so that you can use the overloaded operators to add additional things to a Graph.</p>
<h2><a href="https://pytorch.org/docs/master/compile/technical-overview.html">TorchDynamo</a></h2>
<p>Tracing has limitations in that it can't deal with dynamic control flow and is limited to outputting a single graph at a time, so a better alternative is the new <code>torch.compile()</code> infrastructure where you can output multiple subgraphs in either an aten or torch IR using <code>torch.fx</code>. <a href="https://colab.research.google.com/drive/1Zh-Uo3TcTH8yYJF-LLo5rjlHVMtqvMdf">This tutorial</a> gives more context on how this works.</p>
<h1>The FX IR Container</h1>
<p>Tracing captures an intermediate representation (IR), which is represented as a doubly-linked list of Nodes.</p>
<p>Node is the data structure that represents individual operations within a Graph. For the most part, Nodes represent callsites to various entities, such as operators, methods, and Modules (some exceptions include Nodes that specify function inputs and outputs). Each Node has a function specified by its <code>op</code> property. The Node semantics for each value of <code>op</code> are as follows:</p>
<ul>
<li><code>placeholder</code> represents a function input. The <code>name</code> attribute specifies the name this value will take on. <code>target</code> is similarly the name of the argument. <code>args</code> holds either: 1) nothing, or 2) a single argument denoting the default parameter of the function input. <code>kwargs</code> is don't-care. Placeholders correspond to the function parameters (e.g. <code>x</code>) in the graph printout.</li>
<li><code>get_attr</code> retrieves a parameter from the module hierarchy. <code>name</code> is similarly the name the result of the fetch is assigned to. <code>target</code> is the fully-qualified name of the parameter's position in the module hierarchy. <code>args</code> and <code>kwargs</code> are don't-care</li>
<li><code>call_function</code> applies a free function to some values. <code>name</code> is similarly the name of the value to assign to. <code>target</code> is the function to be applied. <code>args</code> and <code>kwargs</code> represent the arguments to the function, following the Python calling convention</li>
<li><code>call_module</code> applies a module in the module hierarchy's <code>forward()</code> method to given arguments. <code>name</code> is as previous. <code>target</code> is the fully-qualified name of the module in the module hierarchy to call. <code>args</code> and <code>kwargs</code> represent the arguments to invoke the module on, <em>including the self argument</em>.</li>
<li><code>call_method</code> calls a method on a value. <code>name</code> is as similar. <code>target</code> is the string name of the method to apply to the <code>self</code> argument. <code>args</code> and <code>kwargs</code> represent the arguments to invoke the module on, <em>including the self argument</em></li>
<li><code>output</code> contains the output of the traced function in its <code>args[0]</code> attribute. This corresponds to the "return" statement in the Graph printout.</li>
</ul>
<p>To facilitate easier analysis of data dependencies, Nodes have read-only properties <code>input_nodes</code> and <code>users</code>, which specify which Nodes in the Graph are used by this Node and which Nodes use this Node, respectively. Although Nodes are represented as a doubly-linked list, the use-def relationships form an acyclic graph and can be traversed as such.</p>
<h1>Transformation and Codegen</h1>
<p>An invocation of <code>symbolic_traced</code> above requires a valid <code>forward()</code> method to be defined on the Module instance. How does this work? GraphModule actually generates valid Python source code based on the IR it is instantiated with. This can be seen by accessing the code attribute on the GraphModule: <code>print(symbolic_traced.code)</code>.</p>
<p>After tracing, the code given under <a href="#technical-details">Technical Details</a> is represented as follows:</p>
<p><code>python
def forward(self, x):
    param = self.param
    add_1 = x + param;  x = param = None
    linear_1 = self.linear(add_1);  add_1 = None
    clamp_1 = linear_1.clamp(min = 0.0, max = 1.0);  linear_1 = None
    return clamp_1</code></p>
<p>This is the core of why FX is a Python-to-Python translation toolkit. Outside users can treat the results of FX transformations as they would any other <code>nn.Module</code> instance.</p>
<h1>Next steps</h1>
<p>If you're interested in learning more about obtaining fx graphs, which kinds of IRs are available to you and how to execute simple transformations make sure to check out <a href="https://colab.research.google.com/drive/1Zh-Uo3TcTH8yYJF-LLo5rjlHVMtqvMdf">this tutorial</a></p>