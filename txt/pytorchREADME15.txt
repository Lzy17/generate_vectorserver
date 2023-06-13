<p>ATen "native" functions are the modern mechanism for adding operators and
functions to ATen.  Native functions
are declared in <code>native_functions.yaml</code> and have implementations defined
in one of the <code>cpp</code> files in this directory.</p>
<p>Like all ATen methods/functions, native functions are made available
from both ATen's C++ and Python APIs.  In C++, they are made available
either as methods on <code>Tensor</code> (<code>t.mymeth()</code>) and functions in the ATen
namespace (<code>at::myfunc()</code>).  In PyTorch, they are made available as
methods on <code>Variable</code> or as functions on <code>torch._C._FunctionBase</code>.
(It is the user's responsibility to re-export these functions in
a more user-facing module.)</p>
<p>The rest of this document describes how to implement an ATen function.</p>
<h2>Registering a function in <code>native_functions.yaml</code></h2>
<p>Every native function must have an entry in
<code>native_functions.yaml</code>.  The format can be summarized as:</p>
<p><code>- func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -&gt; Return
  variants: function, method
  dispatch:
    CPU: func_cpu
    CUDA: func_cuda</code></p>
<p>Each component is described in more detail below:</p>
<h3><code>func</code></h3>
<p><code>- func: func_name[.overload_name](ArgType arg0[=default], ArgType arg1[=default], ...) -&gt; Return</code></p>
<p>The <code>func</code> entry is a string describing the name of the function and its type
signature.</p>
<p><strong>Argument types.</strong> These types are permissible as ArgType:</p>
<ul>
<li><code>Tensor</code>.  A <code>Tensor</code> argument translates into a C++ argument of type <code>const Tensor&amp;</code>
  (except when the argument is "inplace"; in this case, it is simply <code>Tensor&amp;</code>).
  A trailing <code>?</code>, as in <code>Tensor?</code>, indicates that the tensor argument is optional
  and may be omitted by passing c10::nullopt.  When a function takes multiple
  <code>Tensor</code> arguments, these tensors are assumed to be the same type (e.g.,
  if one argument is a <code>FloatTensor</code>, all other arguments are checked
  to be <code>FloatTensor</code>s).
  <code>Tensor</code> or <code>Tensor?</code> must sometimes be annotated to indicate aliasing and mutability.
  In general annotations can be defined via the following situations:</li>
<li><code>Tensor(a)</code> - <code>a</code> is a set of Tensors that may alias to the same data. The set could have a size of one.</li>
<li><code>Tensor(a!)</code> - members of <code>a</code> may be written to thus mutating the underlying data.</li>
<li><code>Tensor(a! -&gt; a|b)</code> - Tensor is in set <code>a</code>, written to, and after the write is in set <code>a</code> AND <code>b</code>.
  For more details on when and why this needs to happen, please see the section on annotations.</li>
<li><code>Tensor[]</code>.  A <code>Tensor[]</code> argument translates into a C++ argument of type <code>ArrayRef&lt;Tensor&gt;</code>
  (a.k.a. <code>TensorList</code>)</li>
<li><code>int[]</code>.  <code>int[]</code> accepts an optional length specifier, e.g., <code>int[2]</code>, which
  has no effect in C++ but extends our Python bindings to accept a bare number, which will be
  expanded into an appropriately sized list by repeating the number.</li>
<li><code>int</code>. Think about this like a Python int. This is translated into a C++ argument of type <code>int64_t</code>.</li>
<li><code>float</code>. Think about this like a Python <code>float</code>. It is translated into a C++ argument of type <code>double</code>.</li>
<li><code>bool</code></li>
<li><code>str</code>.  It is translated into a C++ argument of non-owning type <code>c10::string_view</code></li>
<li><code>Scalar</code>. <code>Scalar</code> supports binding to any numerical types from Python, including integral types,
  floating point types, and zero dimensional tensors. <code>int</code> and <code>float</code> bind to the corresponding Python
  numerical types. However, you probably don't want to use <code>Scalar</code>;
  <code>float</code> and <code>int</code> argument types should suffice for most algorithms
  (you should only use <code>Scalar</code> if the operator truly may accept either
  type).</li>
<li><code>Generator?</code>, the state for a random number generator,</li>
<li><code>bool[N]</code> (where N is <code>1-4</code>).</li>
<li><code>*</code> is a special sentinel argument, which doesn't translate into an actual
  argument, but indicates that in the Python bindings, any subsequent arguments
  must be specified as keyword arguments (and cannot be provided positionally).</li>
<li><code>?</code> is trailing question mark that annotates an argument to be an optional type. Grep for
  <code>optional</code> to find some example usages. In general, most functions will not need to use
  this, but there are some cases that we want to use optional for the different types:<ul>
<li>You want to pass a <code>None</code> to an ATen function/method from Python and handle the
  None type on the C++ side. For example, <code>clamp(Tensor self, Scalar? min=None, Scalar? max=None)</code>
  can take <code>None</code> for its <code>min</code> and <code>max</code> parameter, but does not dispatch to different
  backends if one of the parameters is <code>None</code>. Optional type can accept a <code>None</code> type
  (<code>nullopt</code> in C++) from Python and use the <a href="https://en.cppreference.com/w/cpp/utility/optional">C++ Optional class</a> to interact with the parameters.</li>
<li>You want a default value, which is fine in Python, but would cause ambiguity in C++.
  For example, <code>norm(Tensor self, Scalar p=2, int dim, bool keepdim=False)</code> would
  cause ambiguity in C++ since its default args must be adjacent (<code>p</code> could not
  have a default value when <code>dim</code> does not). Therefore, we need to make <code>p</code> as a
  optional Scalar, and make <code>p=2</code> when <code>p</code> is not passed in (nullopt).</li>
<li>You want a value to default to the same value as another argument (this cannot be
  expressed in C++ default arguments).</li>
</ul>
</li>
</ul>
<p>Functions with no tensor inputs are called <em>factory functions</em>, and
are handled specially by code generation.  If your function is behaving
differently than another example, check first and see if one is a
factory while another is not. In some rare cases, factory function might have a
tensor argument. In this case mark it with <code>category_override: factory</code>
explicitly.</p>
<p><strong>Argument names.</strong> Argument names are meaningful; downstream binding code may make use of the specific
argument name you provide, and a rename of an argument name is considered a BC-breaking
change (e.g., you will probably need to update <code>tools/autograd/derivatives.yaml</code> at
least, and it may affect Python keyword arguments). For more details please see the section on <code>variants</code>.</p>
<p>As a convention we use 'out' to indicate an output argument. This aligns with the
Python bindings. Even if a function might not be used in the Python bindings, we
still advise to follow this convention. Check the generated code when making a change
to make sure you're not breaking the API when renaming an argument name of an
existing function.</p>
<p><strong>Defaults.</strong> Any suffix of arguments can have a default value defined;
these default values translate into C++/Python default values which
are applied when those positional arguments are not specified.</p>
<p>Here are the supported default values:</p>
<ul>
<li>Numbers (e.g., <code>0</code> or <code>5.0</code> for <code>int</code>, <code>float</code> and <code>int[]</code>
  with an explicit length (e.g., <code>int[2]</code>)--in the case of <code>int[]</code>
  a number is replicated to fill the length (e.g., <code>int[2] x=2</code>
  is equivalent to <code>int[2] x=[2,2]</code>).</li>
<li>Lists of numbers (e.g., <code>[0, 0]</code>) for <code>IntList</code>.</li>
<li>Booleans (e.g., <code>True</code>) for <code>bool</code>.</li>
<li>Empty initializer lists (e.g., <code>[]</code>) for <code>Tensor</code> (this implicitly changes
  a <code>Tensor</code> argument to accept undefined tensors).</li>
<li><code>None</code> for pointer types (e.g., <code>Generator?</code>)</li>
</ul>
<p><strong>Returns.</strong> The following are permissible on Return:</p>
<p>Non-tuple return:
<code>ReturnType [retarg0]</code></p>
<p>Tuple return:
<code>(ReturnType [retarg0], ReturnType [retarg1], ...)</code></p>
<p>The following are permissible on ReturnType:
- <code>Tensor</code> and <code>Tensor[]</code>, which translate into the C++ types <code>Tensor</code> and <code>std::vector&lt;Tensor&gt;</code>,
  respectively (unless the operation is in-place, in which case the return type
  is <code>Tensor&amp;</code>.
- A tuple of any number of <code>Tensor</code>, e.g., <code>(Tensor, Tensor)</code>, translating into
  the C++ <code>std::tuple&lt;Tensor, Tensor&gt;</code>.</p>
<p>If you need a type that is not listed in this list, it may be possible to extend ATen's
code generation to support it.  ATen's philosophy on types to support is that it supports
only simple, universal types, as well as a handful of fundamental Tensor structures
(e.g., <code>Tensor</code> and <code>Generator?</code>), because these types can be easily ported to any language
bound to ATen (in practice, C++ and Python.)</p>
<p>Return also supports specifying (optional) return argument names. These serve
two functions:</p>
<ul>
<li>
<p>They let you easily write derivatives in terms of return arguments in
  <code>tools/autograd/derivatives.yaml</code></p>
</li>
<li>
<p>They correspond to the named field the output can be referred to from
  Python.  (This means that changing a return argument name is
  BC-breaking, be careful!)</p>
</li>
</ul>
<p>Note that argument type modifiers such as defaults and optional are not currently supported on Return.</p>
<p><strong>Overloads.</strong> You can register multiple functions with the same name and different
function signatures if you give them unique overload names. An overload name
is specified after the function name, separated by a dot.</p>
<p>Overload names do not have to be globally unique, but must be unique in the set
of all overloads for the same function. Overload names cannot be changed for
backwards compatibility reasons. Please try to make overload names semantically
meaningful. An overload name that just enumerates all the argument types isn't
helpful. In many cases, a semantic name is clear from what the overload is doing
differently. As a fallback, you can use the name or type of the first differing
argument as an overload name.</p>
<p>If you add a new overload to an existing function, please leave the existing
overload names as they are (for backwards compatibility), but give the new
overload a new, unique name.  Although overload names are not directly
used by the Python or C++ APIs, they are public API surface for external
backends (who register to specific overload names) and deployed mobile
models (which use overload names as part of the serialization format.)</p>
<p>Not specifying an overload name is equivalent to specifying an empty overload
name. If you add a new function with multiple overloads, give them unique
overload names, at most one overload is allowed to have an empty overload name.</p>
<p>The declarations also support the following attributes.</p>
<p><strong>Namespaces.</strong> User can register operators in different namespaces than <code>aten</code>, by simply putting custom namespaces before the function name. Currently nested namespace is not supported for function name. If not specified, all the functions will be registered in <code>aten</code> namespace.</p>
<p>For example, suppose we are registering <code>my_op</code> into <code>custom</code> namespace, we can have:
<code>- func: custom::my_op(Tensor(a) self, ...) -&gt; Tensor(a)
  variants: function, method
  dispatch:
    CPU: my_op_cpu
    CUDA: my_op_cuda</code></p>
<p>Note that we have a one-off <code>TORCH_LIBRARY</code> APIs to achieve the same goal of registering an operator in a custom namespace. Comparing with that API, having custom namespace in <code>native_functions.yaml</code> is useful in cases where the function does not really belong to ATen but is also widely used and it is preferred to have a shared place to register it.</p>
<h3><code>variants</code></h3>
<p><code>variants: function, method</code></p>
<p>Controls whether Tensor method (<code>t.foo()</code>) or namespace Function (<code>at::foo()</code>) is
generated as a result of this declaration.  If the declaration is a method,
you must have an argument <code>Tensor self</code> at some position in the method;
in the method variant this argument will be elided from the argument
list.  For example, given the declaration <code>where(BoolTensor cond, Tensor self, Tensor other)</code>,
this generates the function <code>at::where(cond, self, other)</code> and the method
<code>self.where(cond, other)</code>.</p>
<p>By default, ATen generates only the function variant for a native function.
When should you also generate a method variant? Tensor operations as methods
are appropriate for "core" Tensor operations (e.g., add, sub, etc.), but not for
more complicated neural network layers (e.g., <code>conv2d</code>) and internal functions
designed specifically for binding (e.g., <code>cudnn_convolution</code>).</p>
<p>As we progress along our schema unification of the <code>func</code> schema with the JIT
signature schema, we must introduce features that allow us to increase compliance.
One of these features are Tensor annotations. As of now we use naming conventions
to indicate whether an argument of a function is going to be mutated and returned.</p>
<h3><code>annotations</code></h3>
<p>There are two typical situations in which we mutate the memory of an argument in the Python
frontend:
a) For an inplace operations such as <code>self.abs_()</code>
b) for a function with an output keyword argument such as <code>torch.abs(input, out=None)</code>.</p>
<p>In order to provide implementations for these Python functions the legacy schema
requires C++ implementations for three situations <code>abs(Tensor self)  -&gt; Tensor</code>,
<code>abs_(Tensor self) -&gt; Tensor</code> and <code>abs_out(Tensor out, Tensor self) -&gt; Tensor</code>.</p>
<p>Now, as we move towards the unification, we start to use a different syntax to represent
this by using annotations. In the end we still translate to the legacy schema for the downstream
consumers such as the C++ code generation, but this will soon change.</p>
<p>If two Tensors carry the same annotation, they both <em>may</em> represent the same memory.
A write annotation, as indicated by an exclamation mark, indicates that they both <em>may</em>
also be written to.</p>
<p>Let's revisit the previous native function declarations and see the conventions of adding annotations.
  - <code>abs(Tensor self) -&gt; Tensor</code> stays the same as it will always allocate new memory.
  - <code>abs_(Tensor(a!) self) -&gt; Tensor(a!)</code>
    <code>self</code> may be written to and returned. Further, the annotation indicates that the return value
    may alias the input. This indicates an inplace function and by convention ends in a single '_'.
  - <code>abs(Tensor self, *, Tensor(a!) out) -&gt; Tensor(a!)</code>
    In the Python frontend <code>out</code> can be passed as a keyword argument and may be written to.
    In this case it indicates the schema for a function that must accept <code>out</code> as this does not
    provide a default argument. The idea behind representing this as a optional argument is to
    document the intended usage. This maps to the legacy <code>abs_out(Tensor out, Tensor self) -&gt; Tensor</code>.
    As with the legacy <code>_out</code> function you must call the argument <code>Tensor out</code> or <code>Tensor out0</code>,
    <code>Tensor out1</code> in the context of multiple arguments.</p>
<p>There is also another situation in which we use annotations, namely views.
  - <code>transpose(Tensor(a) self, int dim0, int dim1) -&gt; Tensor(a)</code>
    An alias to the memory represented by <code>self</code> may be also returned, however it is not mutated.</p>
<p>When a Tensor views are contained in a Tensor list, we need to represent that the output list
contains Tensors that alias the input.
  - <code>func: chunk(Tensor(a -&gt; *) self, int chunks, int dim=0) -&gt; Tensor(a)[]</code>
We assume lists contain memory which aliases the heap, so in order to correctly set up the aliasing
relationship between the output and input, we annotate that the input Tensor enters the wildcard set <code>(a -&gt; *)</code>.
For more details, see the JIT <a href="https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#aliasing-and-mutation-annotations-in-functionschema">README</a>.</p>
<p>We have some asserts to check whether a developer uses these annotations correctly and throw asserts
if she doesn't. For example, any out function must use the <code>(a!)</code> annotation as described above.
 If this causes a lot of confusion please add @cpuhrsch to your PR.</p>
<h3><code>dispatch</code></h3>
<p><code>dispatch:
    CPU: func_cpu
    CUDA: func_cuda</code></p>
<p>This specifies the actual name of the function you want to dispatch to, so you
can dispatch to different functions depending on which backend the passed tensors
belong to.  Notice that custom namespaces is supported on these names, it's useful when the native function listed lives in a namespace other than the default <code>at::native</code>. Currently we support nested namespace with maximum level of 2. For example:
<code>dispatch:
    CPU: custom::ns::func_cpu</code>
The example above hinted the native function can be found under <code>custom::ns::native</code> namespace (the trailing <code>::native</code> is added automatically).</p>
<p>If the dispatch table is omitted, we assume a default dispatch
table:</p>
<p>```</p>
<h1>overload is ignored</h1>
<p>func: func.overload(...) -&gt; ...
dispatch:
    CompositeImplicitAutograd: func</p>
<h1>overload is ignored, but out functions get suffixed with _out in their name</h1>
<h1>(NB: no out functions in PyTorch today actually support autograd, but if they</h1>
<h1>did, you could call them here and autograd would be inferred)</h1>
<p>func: func.out_overload(...) -&gt; ...
dispatch:
    CompositeImplicitAutograd: func_out
```</p>
<p>If two backends have the same dispatch function, you can write <code>CPU, CUDA: func</code>
to reuse the same function name in both cases.</p>
<p>Available backend options can be found by searching <code>dispatch_keys</code> in
<a href="https://github.com/pytorch/pytorch/blob/master/torchgen/gen.py">codegen</a>.
There are also three special "generic" backends:</p>
<ul>
<li>
<p><code>CompositeExplicitAutograd</code> (previously known as <code>DefaultBackend</code>):
    implementations of kernels that work for all backends, but require an
    explicit definition of backward function in <code>derivatives.yaml</code> to support autograd.
    The most typical use of this key are for delegating functions; i.e.,
    functions that do a very small amount of work and then delegate to another
    operator to do the actual heavy lifting.  Under the hood, registering a
    kernel to <code>CompositeExplicitAutograd</code> is equivalent to registering that
    kernel to every backend (e.g., <code>CPU, CUDA</code>). Note: kernels which call
    DispatchStub should NOT be registered as CompositeExplicitAutograd, as
    DispatchStub only works for <code>CPU, CUDA</code>)</p>
</li>
<li>
<p><code>CompositeExplicitAutogradNonFunctional</code>:
    Similar to CompositeExplicitAutograd, but this key should be used if:
    (1) Your kernel is written for a non-aliasing operator.
    (2) <em>and</em> it calls internally into an aliasing operator.
    An example of this is select_backward, which is non-aliasing, but decomposes into select.
    We would like to distinguish between "ordinary" CompositeExplicitAutograd kernels
    and these kernels, because some backends would not like
    to decompose an non-aliasing op into an aliasing op.
    LazyTensor + XLA are the two current examples of this - since they operate on a functional IR,
    they would prefer to directly implement a non-aliasing operator with their own kernel,
    instead of using a decomposition that results in more aliasing operators.</p>
</li>
<li>
<p><code>CompositeImplicitAutograd</code> (previously known as <code>Math</code>): implementations of
    kernels that work for all backends, and also can implicitly support autograd,
    because all of the operations it calls support autograd.  Direct use of
    this key should be rare: if you provide no dispatch table, we default to
    registering your kernel as <code>CompositeImplicitAutograd</code>.  Explicitly adding
    this key to an existing dispatch table may be useful if you have specialized
    CPU and CUDA implementations, but you might want to provide a fallback
    lowering for external backends that may not have a specialized
    implementation.</p>
</li>
</ul>
<p>Functions registered to composite backends should work for any backend, if the
nested functions they call work for those backends.</p>
<p>For example, suppose <code>my_op</code> can be implemented in the following way:</p>
<p><code>at::Tensor my_op(const Tensor&amp; self, const Tensor&amp; other) {
  return self + 2 * other;
}</code></p>
<p>If we already know inference kernels and derivative formulas for operators <code>+</code> and <code>*</code> in our system,
you can just register <code>my_op</code> to <code>CompositeImplicitAutograd</code> and both inference &amp; autograd will just work.
Although it seems we only write down the inference formula here, PyTorch autograd system would correctly
set up the backward for <code>my_op</code> using the chain formula and derivatives of <code>+</code> &amp; <code>*</code> operators.
In other words <code>d_out/d_self = 1; d_out/d_other = 2</code> can be derived automatically from
the <code>my_op</code> inference kernel. Of course if we don't have derivative formula defined for either <code>+</code> or <code>*</code>,
backward of <code>my_op</code> can no longer be derived automatically.</p>
<p>Whether to use implicit or explicit autograd for your kernel can be decided by the following steps:
1. If you can, always start with a <code>CompositeImplicitAutograd</code> kernel that's composable from existing operators.
2. If you don't want to use the derived gradient formula from <code>CompositeImplicitAutograd</code> kernel for autograd, either to
   get better performance or better numerical stability, you should register the kernel with <code>CompositeExplicitAutograd</code>
   so that it's only used in inference.
   Later for autograd, depending on whether your autograd kernel works for all backends or not,
   you can put them in alias <code>Autograd</code> or specific keys like <code>AutogradCPU</code>.
3. If you prefer to write backend-specific kernels, use reserved dispatch keys for your backend instead,
   e.g. <code>CPU/AutogradCPU</code>.</p>
<p><strong>Important</strong>: because a <code>CompositeImplicitAutograd</code> kernel is implicitly registered for ops with no <code>dispatch:</code> section,
when you add a backend-specific kernel (and hence a <code>dispatch:</code> section) to one of these, you <strong>must</strong> also
add a <code>CompositeImplicitAutograd:</code> entry that names the old kernel implementation (it's named after the op, with _<overload name>
added if applicable), so that it's still available for other backends to use.</p>
<p>If you implemented a native function in C++ and want to find out which dispatch keyword
should be used in native_functions.yaml, please <a href="#choosing-the-right-dispatch-keyword">follow steps in dispatch keywords</a></p>
<h3>Composite Compliance</h3>
<p>Definition: a "composite function" is an Operator registered as
CompositeImplicitAutograd or a (Python or C++) function that consists of PyTorch
operations. Examples of the latter include backward formulas and forward-mode AD formulas.</p>
<p>Composite functions defined in the PyTorch library MUST work for most, if not
all, backends/subclasses. This means that we impose a set of constraints that make it more
difficult to write composite functions inside PyTorch library code than users
writing PyTorch code.</p>
<p>If you wish to do something that is banned (you may wish to do this for perf
reasons), please write a backwards formula for your function so it is no longer
hide parts of the function in a new aten operator that is not CompositeImplicitAutograd.</p>
<p>Composite functions may not:
- call <code>resize_</code> or moral equivalents. These are tricky to handle for
many backends, like vmap and meta.
- call <code>out=</code> operations. These are impossible to handle for vmap and can cause
dispatch-to-python objects to lose their subclassing.
- Change the metadata of a Tensor without performing dispatches. Examples of these
operations are directly accessing the TensorImpl API to modify the
sizes/strides/metadata of a Tensor.
- In the same vein as the last point, <code>data_ptr</code> access or <code>item</code> access are not
allowed. These operations do not go through the dispatcher.
- <code>copy_</code> is a marginal case. If you're able to rewrite your operation without
<code>copy_</code> you should definitely do so; this should be trivial if you're not copy-ing
into a view. Otherwise, it is fine to leave the code as-is.</p>
<p>We have CompositeImplicitAutograd compliance tests in <code>test/test_ops.py</code>. These
tests aren't perfect (it's pretty difficult to check for all of the above) so if
something looks wrong please shout.</p>
<h3><code>device_guard</code></h3>
<p><code>device_guard: False</code></p>
<p>By default, ATen code generation will generate a DeviceGuard invocation,
which will ensure that kernel code will run with the current device set
to match the device of the first Tensor argument (or first tensor of
the first Tensor[] argument, if the function takes a list of tensors).
For the most part, this means kernel authors do not have to worry about
setting devices.</p>
<p>However, in some cases, setting the device is unnecessary, because,
e.g., you call a function already manages device guard setting, or
you're a function that simply does not interact with any devices. In
that case, code generation of the device guard can be disabled by adding
<code>device_guard: False</code> to your function definition.</p>
<h3><code>device_check</code></h3>
<p><code>device_check: NoCheck</code></p>
<p>By default, ATen code generation will generate device check,
which will ensure all the tensor parameters passed to kernel are
on the same device.</p>
<p>However, in some cases, checking the device is unnecessary, because,
e.g., you call a function allows to work on multiple devices.
In that case, code generation of the device check can be disabled by adding
<code>device_check: NoCheck</code> to your function definition.</p>
<h3><code>manual_kernel_registration</code></h3>
<p><code>manual_kernel_registration: True</code></p>
<p>With this flag set, we will not generate code to automatically register the C++ operator implementation
to TypeDefault (catchAll dispatch key) with the dispatcher.
It doesn't make sense to have both <code>dispatch</code> section and <code>manual_kernel_registration: True</code> for the same op.
You can find the manual registrations in torch/csrc/autograd/VariableTypeManual.cpp.
Currently ops have this field set to True should match <code>MANUAL_CATCHALL</code> in tools/autograd/gen_variable_type.py
(It can be a superset of <code>MANUAL_CATCHALL</code> but we don't have a use case for it).
This field should only be used rarely.</p>
<h3><code>use_const_ref_for_mutable_tensors</code></h3>
<p><code>use_const_ref_for_mutable_tensors: True</code></p>
<p>With this flag set, we will generate arguments for Tensors whose underlying data may change as
<code>const Tensor&amp;</code> (or similar), just like we would for other Tensors. Previously, we generated these
as <code>Tensor &amp;</code>, which 1) allowed changing which <code>TensorImpl</code> the <code>Tensor</code> itself referred to and 2)
was not necessary to allow the underlying data to change. (This was like using <code>T * const</code> when we
wanted <code>const T*</code>.)</p>
<h3><code>autogen</code></h3>
<p><code>- func: my_op_(Tensor(a!) self) -&gt; Tensor(a!)
...
  autogen: my_op, my_op.out</code></p>
<p><code>autogen</code> keyword is being used to specify which native function the codegen system should generate
implementations for.
* For an in-place variant of a native function (op name ends with an <code>_</code>), we will generate a functional
variant and an out= variant.
* If a functional variant is given, we generate an out= variant.
* We don't support <code>autogen</code> for view ops, ops that bypass the dispatcher as well as composite ops.</p>
<p>We also generate kernels for generated ops, which merely copy and return the result from the base ops.
These generated kernels can be found in <code>&lt;gen-out&gt;/aten/src/ATen/CompositeViewCopyKernels.cpp</code>.</p>
<p>Also notice that for new operators being added to <code>native_functions.yaml</code>, if they satisfy the requirements
mentioned above, they should include <code>autogen</code> keyword, since functionalization depends on it. We will
enforce this in codegen.</p>
<h2>Writing an implementation in C++</h2>
<p>Implementations of native functions go in an appropriate C++ file in the
<code>native/</code> directory (they are organized roughly by topic, but there is no
semantic meaning to their organization aside for the <code>cuda</code> directory,
which is the only place the build system knows how to build <code>cu</code> files.)
To write a native function, you only need to write a C++
implementation (no header necessary) with a matching signature to
the generated header from the ATen metadata.  There are many
simple native functions; take a look at some of them to see what to do.</p>
<p>Although writing an ATen function is mostly writing the algorithm you want
to implement, there are some less obvious details you should also consider.</p>
<h3>Will your function be automatically differentiable?</h3>
<p>If you are writing a pair of functions <code>foo</code> and <code>foo_backward</code>, with
the intent that <code>foo_backward</code> implements the derivative of <code>foo</code>, then
your implementation of <code>foo</code> is probably not automatically differentiable:
it might make use of functions like <code>data_ptr()</code> or it dispatches differently
depending on if it's operating on CPU or CUDA tensors.  Once you write these two functions,
you will have to write an entry correlating them together in
<code>tools/autograd/derivatives.yaml</code>.</p>
<p>However, in some situations, you can write a function in ATen and it
will be automatically differentiated! This can be the case if the function implementation
only calls other operations which are themselves differentiable.  In this
case, you don't have to write an entry in <code>tools/autograd/derivatives.yaml</code>.</p>
<h3>Choosing the right dispatch keyword</h3>
<p>After writing a native function in C++, it's important to think about which dispatch keyword
to use in native_functions.yaml as it gives the dispatcher information about backend and autograd support
of the implementation.</p>
<p>Here're steps to follow to decide the right dispatch keyword:</p>
<ol>
<li>
<p>Think about inference: does your kernel work for all backends?</p>
<ul>
<li>No: you're likely providing different kernels for different backends, e.g.
  backend-dependent logic is used in the implementation or it's implemented through DispatchStub.
  DispatchStub only support a backend if you explicitly provide a kernel through <code>REGISTER_DISPATCH</code>.
  Typically it only supports a few in-tree backends like CPU, CUDA, QuantizedCPU etc but not
  out-of-tree backends like XLA.
  Write a dispatch section, enumerate all supported backends and point them to the implementations.
  <code>dispatch:
    CPU: kernel_cpu
    CUDA: kernel_cuda
    QuantizedCPU: kernel_quantized_cpu</code></li>
</ul>
<p>You're done. Now this op will be called in <code>CPU/CUDA/QuantizedCPU</code> backend inference!</p>
<p>Note: to support training, you're required to write a formula in
  derivatives.yaml since your backend implementations don't support autograd.</p>
<ul>
<li>Yes: you're likely calling other <code>at::</code> ops in the implementation. Go to step 2.</li>
</ul>
</li>
<li>
<p>Think about training: does your kernel support autograd? <a href="#will-your-function-be-automatically-differentiable">check autograd support</a></p>
<ul>
<li>
<p>Yes: in other words, you're providing a <code>CompositeImplicitAutograd</code> kernel which supports both inference and autograd.
  To use autograd support for training, simply skip adding a dispatch
  section and you're done. This will allow this op to be correctly
  registered for both inference and training.</p>
</li>
<li>
<p>Yes, but you still want to provide a numerically stable gradient formula instead of using autograd, write
  <code>dispatch:
    CompositeExplicitAutograd: kernel</code></p>
</li>
</ul>
<p>You're done. This op will be called in inference for all backends.</p>
<p>Note: to support training you're required to add an autograd formula,
  or it'll error out in backward pass when calling with a Tensor has requires_grad=True.</p>
<ul>
<li>No: ops in this category are mainly using <code>_out</code> boilerplate where its out version doesn't have a derivative
  formula defined. For example:
  <code>Tensor&amp; sign_out(Tensor&amp; result, const Tensor&amp; self) { return unary_op_impl_out(result, self, sign_stub); }
  Tensor sign(const Tensor&amp; self) { return unary_op_impl(self, at::sign_out); }
  Tensor&amp; sign_(Tensor&amp; self) { return unary_op_impl_(self, at::sign_out); }</code></li>
</ul>
<p><code>sign_out</code> uses DispatchStub so the supported backends are enumerated in its dispatch section.
  For <code>sign</code> and <code>sign_</code>, write
  <code>dispatch:
    CompositeExplicitAutograd: kernel</code></p>
<p>You're done. This op will be called in inference for all backends.</p>
<p>Note: to support training you're required to add an autograd formula for <code>sign</code>,
  or it'll error out in backward pass when calling with a Tensor has requires_grad=True.</p>
<p>Note: current plan on record for ops using this boilerplate is to replace <code>at::</code> with <code>at::native</code> in
  the implementations and add dispatch section with device keywords instead.
3. Validate the computed dispatch table matches what you want. You can use <code>PythonDispatcher</code> provided in
<a href="https://github.com/pytorch/pytorch/blob/master/torch/_python_dispatcher.py">torch/_python_dispatcher.py</a>.
It shows for a certain operator, what the computed dispatch table looks like after your registrations.</p>
<p><code>dispatcher = PythonDispatcher()
dispatcher.register(["CPU", "XLA", "AutogradCPU", "CompositeImplicitAutograd"])
print(dispatcher.dispatchTable()) # Tells you exactly which kernel is used for certain backend.</code></p>
</li>
<li>
<p>TODO: AutogradCPUOrCUDA</p>
</li>
</ol>
<p>Note that in native_functions.yaml you can mix using backend keywords and alias keywords above for one op:
  - direct registration to backend always has higher precedence than alias
  - DO NOT provide multiple alias keywords to the same op: alias keywords have precedence <code>CompositeExplicitAutograd &gt; CompositeImplicitAutograd</code>,
    e.g. adding both <code>CompositeImplicitAutograd</code> and <code>CompositeExplicitAutograd</code> kernels for one op will completely ignore <code>CompositeImplicitAutograd</code> kernel for
    both inference and training. Thus this will trigger an error when native_functions.yaml is parsed.</p>
<h3>Will this function be exposed to python? What are the namespaces?</h3>
<p>We don't generate python bindings for all functions. There're certain patterns in function
name that we skip in python binding generation, e.g. <code>*_backward</code>. Check
<code>tools/autograd/gen_python_functions.py</code> for the latest rules.</p>
<p>The generated bindings are either exposed as methods on python_variable or functions on
the torch._C._nn (marked with <code>python_module: nn</code>),
torch._C._fft (marked with <code>python_module: fft</code>),
torch._C._linalg (marked with <code>python_module: linalg</code>) objects,
torch._C._sparse (marked with <code>python_module: sparse</code>) objects,
torch._C._special (marked with <code>python_module: special</code>) objects,
or torch._C._nested (marked with <code>python_module: nested</code>) objects.</p>
<h3>Undefined tensor conventions</h3>
<p>By default, <code>Tensor</code> arguments to ATen functions are always defined, unless
you explicitly specified that an undefined tensor was permissible by writing
<code>Tensor?</code> or <code>Tensor? x=[]</code>, the latter one is needed when you have to assign
a default value in C++ (e.g. in the middle of other parameters with default values).</p>
<p>The rules for returning undefined Tensors are a bit more subtle, but there
is only one case you have to remember:</p>
<ul>
<li>
<p>If the function in question is a backward function which accepts a
  <code>std::array&lt;bool,N&gt; output_mask</code> argument, you MUST return an undefined
  <code>Tensor</code> at every tuple position <code>i</code> for which <code>output_mask[i]</code> is false, otherwise</p>
</li>
<li>
<p>You MUST NOT return an undefined tensor.</p>
</li>
</ul>
<p>The most common situations where you might be tempted to return undefined tensors
are when:</p>
<ul>
<li>
<p>You have a forward function that may return a buffer if training is enabled, but does not
  return the buffer in inference mode.  In this case, just return an appropriately
  typed zero-size tensor.</p>
</li>
<li>
<p>You have a backward function where the gradient for an input is zero.  In this case, you
  are expected to create a zero-filled tensor of appropriate size to return for this input.
  To get the shape, it may be helpful to take a <code>TensorGeometry</code> of the input to use.</p>
</li>
</ul>
<h3>Debugging tips</h3>
<p>If you build ATen and get a linker error, that probably means you copy-pasted
the C++ definition of your function incorrectly.  Double check your <code>Tensor</code>
arguments, and make sure you wrote <code>const Tensor&amp;</code> in your signature.</p>