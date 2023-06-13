<h1>Howto: Writing PyTorch &amp; Caffe2 Operators</h1>
<p>So you want to write a new operator or a new kernel for an existing operator. How do you do that and what API should you use? So glad you asked.</p>
<h2>native_functions.yaml vs custom operators</h2>
<p>All operators that are part of the public API of PyTorch are defined in <code>native_functions.yaml</code>. Just add an entry there and write the corresponding C++ kernel function. It’s very easy and there is a good introduction at https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/README.md.</p>
<h3>So when should you <strong>not</strong> use <code>native_functions.yaml</code>?</h3>
<p>There’s four main use cases</p>
<ul>
<li>You’re writing a new operator that isn’t supposed to be part of the public PyTorch API.</li>
<li>You’re writing a new operator but don’t want to change the core pytorch code base, say you’re developing a shared library with operators.</li>
<li>You’re writing a C++ extension for PyTorch or you’re using inline c++ in your .py model files.</li>
<li>You’re writing a backend library like XLA or ORT that adds new kernels to all operators defined in <code>native_functions.yaml</code>.</li>
</ul>
<p>For these use cases, the custom operator API is the better solution.</p>
<h3>What is the price for using the custom operator API instead of <code>native_functions.yaml</code>?</h3>
<p>If you’re just using the custom operator API to add new kernels for existing operators (e.g. the XLA/ORT example above), then you’re fine and don’t pay any price. If, however, you define a new operator purely using the custom op API, i.e. your operator never shows up in <code>native_functions.yaml</code>, then you need to be aware of a few caveats.</p>
<ul>
<li>It will not get a C++ API generated. There will not be <code>Tensor::your_op()</code> methods or <code>at::your_op()</code> functions to call your operator.</li>
<li>The API for calling the operator from Python looks a little bit different. It needs to be called through <code>torch.ops.your_op()</code> instead of <code>torch._C</code>.</li>
<li>Setting up autograd for custom operators is harder. You don’t get it automatically but need to use <code>torch::autograd::Function</code> to implement autograd (<a href="https://github.com/pytorch/pytorch/blob/d762ad09df7f5808196b0e2e417b6592e0d30a30/test/cpp/api/autograd.cpp#L126-L152">example</a>). Note also that <code>torch::autograd::Function</code> does not work together with dispatch yet, so if you have different kernels for different backends (say CPU and CUDA), you need to manually write if/else statements for that.</li>
</ul>
<h2>Writing custom operators</h2>
<p>So, you’ve read all above but still want to use the custom operator API? Great. Here’s how you do it.</p>
<p>There's two ways you can write kernels for a PyTorch operator. You can write them as functions or as lambdas.</p>
<h3>As functions</h3>
<p>This is probably the most simple way to write an operator. Just write a kernel function and register it with the PyTorch operator library.</p>
<p>```
namespace { Tensor my_kernel_cpu(const Tensor&amp; a, const Tensor&amp; b) {...} }</p>
<p>static auto registry = torch::RegisterOperators()
   .op("my_namespace::my_op",  torch::RegisterOperators::options()
       .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(CPU()));
```</p>
<p>It is recommended to put your kernel into an anonymous namespace because that allows for better linker optimizations and smaller binary size.
The dispatch key argument (i.e. <code>CPU()</code>) takes care that this kernel is only called for tensors from the CPU backend, more on that below.</p>
<h3>As lambdas</h3>
<p>Very short and simple kernels can be written as lambdas directly in the registration call:</p>
<p><code>static auto registry = torch::RegisterOperators()
    .op("my_namespace::my_op", torch::RegisterOperators::options()
        .kernel(CPU(), [] (const Tensor&amp; a) -&gt; Tensor{...}));</code></p>
<p>These lambdas must be stateless, i.e. not have a closure. The registration will fail if the lambda has a closure.</p>
<h3>Catch-all kernels</h3>
<p>You can register catch-all kernels that are called for every backend. This disables dispatch for this operator and just always calls into the kernel you provide. You cannot combine catch-all kernels and regular device-bound kernels for the same operator.</p>
<p>```
namespace { Tensor my_kernel_fallback(Tensor a, Tensor b) {...} }</p>
<p>static auto registry = torch::RegisterOperators()
   .op("my_namespace::my_op", torch::RegisterOperators::options()
       .catchAllKernel<decltype(my_kernel_fallback), &my_kernel_fallback>());
```</p>
<p>The other ways of specifying kernels mentioned above (as functions, functors or lambdas) also work with <code>catchAllKernel()</code>.</p>
<h3>Syntactic Sugar</h3>
<p>You can use the following syntactic sugar to define a catch-all kernel function more easily:</p>
<p>```
namespace { Tensor my_kernel_cpu(const Tensor&amp; a, const Tensor&amp; b) {...}</p>
<p>static auto registry = torch::RegisterOperators()
 .op("my_namespace::my_op", &amp;my_kernel_cpu);
```</p>
<p>or for lambdas:</p>
<p><code>static auto registry = torch::RegisterOperators()
 .op("my_namespace::my_op", [] (Tensor a, Tensor b) {...});</code></p>
<h2>Chaining</h2>
<p>Multiple operator registrations can be chained into the same registry by calling <code>.op()</code> multiple times:</p>
<p><code>static auto registry = torch::RegisterOperators()
    .op("my_namespace::my_op_1", torch::RegisterOperators::options()
        .kernel&lt;MyKernel1&gt;(CPU()))
    .op("my_namespace::my_op_2", torch::RegisterOperators::options()
        .kernel&lt;MyKernel2&gt;(CPU()));</code></p>
<h2>Multiple Backends</h2>
<p>You can register different kernels for the same operator for different backends.</p>
<p>```
namespace {
Tensor my_kernel_cpu(const Tensor&amp; a, const Tensor&amp; b) {...}
Tensor my_kernel_cuda(const Tensor&amp; a, const Tensor&amp; b) {...}
}</p>
<p>static auto registry = torch::RegisterOperators()
   .op("my_namespace::my_op",  torch::RegisterOperators::options()
       .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(CPU()))
   .op("my_namespace::my_op",  torch::RegisterOperators::options()
       .kernel<decltype(my_kernel_cuda), &my_kernel_cuda>(CUDA()));
```</p>
<p>Note that here, the CPU and CUDA kernel were registered directly next to each other, but that's not necessary. You could even put them into different shared libraries if you want and as long as both are loaded into your process, things will work as you expect.</p>
<h2>The operator schema</h2>
<h3>Explicitly defining the schema</h3>
<p>All examples above automatically inferred the operator schema from the kernel function/lambda. Sometimes, however, you want to specify the schema manually. To specify annotations for example, or default values for arguments (default values will not be inferred from the c++ kernel function), or simply for documentation purposes or to make sure the schema matches your expectations.</p>
<p>```
namespace { Tensor my_kernel_cpu(const Tensor&amp; a, const Tensor&amp; b) {...} }</p>
<p>static auto registry = torch::RegisterOperators()
   .op("my_namespace::my_op(Tensor a, Tensor b) -&gt; Tensor",
       torch::RegisterOperators::options()
         .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(CPU()));
```</p>
<p>Or with annotations:</p>
<p>```
namespace {
    Tensor my_kernel_cpu(const Tensor&amp; a, int64_t b, at::optional<int64_t> c) {...}
}</p>
<p>static auto registry = torch::RegisterOperators()
   .op("my_namespace::my_op(Tensor(a) x, int y = 3, int? z = None) -&gt; Tensor(a|b)",
       torch::RegisterOperators::options()
         .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(CPU()));
```</p>
<p>If the schema is explicitly specified but doesn't match the kernel signature, you will get an error when registering it.</p>
<h3>Multiple outputs</h3>
<p>The kernel function can either return <code>void</code> or a single element like <code>Tensor</code> in the examples above, or it can return multiple values using <code>std::tuple</code> as shown in the following example:</p>
<p>```
namespace {
  std::tuple<Tensor, int64_t, Tensor>
     my_kernel_cpu(const Tensor&amp; a, const Tensor&amp; b, int64_t c) {...}
}</p>
<p>static auto registry = torch::RegisterOperators()
   .op("my_namespace::my_op", torch::RegisterOperators::options()
       .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(CPU()));
```</p>
<h3>Supported Input and output types</h3>
<p>The kernel function can take any of the following types as inputs or outputs:</p>
<ul>
<li><code>at::Tensor</code></li>
<li><code>double</code> (note: <code>float</code> is not supported)</li>
<li><code>int64_t</code> (note: other integer types like <code>int</code>, <code>uint64_t</code>, <code>int32_t</code>, <code>...</code> are not supported)</li>
<li><code>bool</code></li>
<li><code>c10::string_view</code></li>
<li><code>at::Scalar</code> (this is a type that can hold either an integer or a floating point value)</li>
<li><code>at::optional&lt;T&gt;</code> with T being any type from the list above</li>
</ul>
<p>The kernel function can take and return list inputs by using <code>torch::List&lt;T&gt;</code>. <code>T</code> must be one of the supported types from above excluding <code>at::Scalar</code>.</p>
<p>The kernel function can take and return dicts by using <code>torch::Dict&lt;Key, Value&gt;</code>. <code>Key</code> must be <code>int64_t</code>, <code>c10::string_view</code>, <code>double</code> or <code>bool</code>, and <code>Value</code> must be from the list of supported types above excluding <code>at::Scalar</code>.</p>
<p>When taken as input, any of these types can be taken by value (i.e. <code>Tensor</code>) or by const-reference (i.e. <code>const Tensor&amp;</code>). We recommend taking all arguments by value, even Tensors. They will be moved in, so there is no performance overhead.</p>
<p>If you need another type, it might work but not be officially supported (yet). Please reach out to Sebastian Messmer and we'll see what we can do.</p>
<h3>Overloads</h3>
<p>When multiple kernels are registered for the same operator, they must have the same schema or registration will fail.
<em>Note: This also includes schema properties like annotations or default arguments. If one kernel specifies a schema with annotations or a default argument, all kernels for this operator must do this. Schemas automatically inferred from kernel functions will not have annotations or default arguments. This means to use annotations or default arguments, all kernels for this operator must explicitly specify the schema.</em></p>
<p>If you want to reuse the same operator name for a different schema, you can use overloads. Overloads must be named and the name is appended to the operator name after a dot:</p>
<p>```
namespace {
  Tensor my_kernel_cpu_1(const Tensor&amp; a) {...}
  Tensor my_kernel_cpu_2(const Tensor&amp; a, const Tensor&amp; b) {...}
}</p>
<p>static auto registry = torch::RegisterOperators()
   .op("my_namespace::my_op.overload1(Tensor a) -&gt; Tensor",
       torch::RegisterOperators::options()
         .kernel<decltype(my_kernel_cpu_1), &my_kernel_cpu>(CPU()))
   .op("my_namespace::my_op.overload2(Tensor a, Tensor b) -&gt; Tensor",
       torch::RegisterOperators::options()
         .kernel<decltype(my_kernel_cpu_2), &my_kernel_cpu>(CPU()));
```</p>
<p>Kernels registered for the same overload must have exactly matching schemas, but kernels registered for different overloads are allowed to have different schemas. This also works when different overloads come from different shared libraries.</p>
<h3>Schema-only operators</h3>
<p>You can register an operator without a kernel:</p>
<p><code>static auto registry = torch::RegisterOperators()
   .op("my_namespace::my_op(Tensor a, Tensor b) -&gt; Tensor");</code></p>
<p>In this case, you must explicitly specify the full schema and you must not specify a dispatch key.
This is useful to define the interface of an operator when you don't know a kernel yet. As mentioned above in the “Overloads” section, you will get an error if any kernel registered for this operator has a mismatching signature.</p>
<h2>Calling custom operators</h2>
<h3>From PyTorch/JIT</h3>
<p>All registered operators are automatically available to PyTorch and JIT under <code>torch.ops.XXX</code>. If your operator was <code>my_namespace::my_op</code>, you can call it from python or JIT using <code>torch.ops.my_namespace.my_op(a, b)</code>.</p>
<h3>From caffe2</h3>
<p>Custom operators are not available to the caffe2 frontend by default, but there's a simple macro you can add if you want to make it available. To expose a CPU kernel:</p>
<p><code>// Expose "my_namespace::my_op" custom operator to caffe2.
// In caffe2, the operator will be called "MyCaffe2OperatorName".
C10_EXPORT_C10_OP_TO_CAFFE2_CPU(
    MyCaffe2OperatorName, "my_namespace::my_op")</code></p>
<p>And to expose a CUDA kernel:</p>
<p><code>C10_EXPORT_C10_OP_TO_CAFFE2_CUDA(
    MyCaffe2OperatorName, "my_namespace::my_op")</code></p>
<p>Note that this doesn't autogenerate a caffe2 operator schema for you (yet). If there's need, we might consider adding that in future, but for now you have to write the caffe2 <code>OPERATOR_SCHEMA</code> macro manually if you need it.</p>
<p>Also, there's some requirements on the operator schema for it to be callable from caffe2. Some of these restrictions are just because the functionality isn't implemented. If you have a use case that is blocked by them, please reach out to Sebastian Messmer.</p>
<ul>
<li>There must be either one or more arguments of type <code>Tensor</code>, or one argument of type <code>Tensor[]</code>. You cannot have both <code>Tensor</code> and <code>Tensor[]</code>.</li>
<li>Except for <code>Tensor</code> or <code>Tensor[]</code>, only arguments of type <code>int</code>, <code>double</code> and <code>bool</code> are supported. These can be in any position in the argument list and will be read from the caffe2 operator arguments, based on the argument name in the operator schema.</li>
<li>We do not support lists (<code>int[]</code>, <code>double[]</code> or <code>bool[]</code>) or optionals (<code>int?</code>, <code>double?</code>, <code>bool?</code>) yet.</li>
<li>The operator must return a single <code>Tensor</code> or multiple tensors as in <code>(Tensor, Tensor, Tensor)</code>. It cannot return a list <code>Tensor[]</code>, optional <code>Tensor?</code> or any primitive types.</li>
</ul>