<h1>functorch</h1>
<p><a href="#why-composable-function-transforms"><strong>Why functorch?</strong></a>
| <a href="#install"><strong>Install guide</strong></a>
| <a href="#what-are-the-transforms"><strong>Transformations</strong></a>
| <a href="#documentation"><strong>Documentation</strong></a>
| <a href="#future-plans"><strong>Future Plans</strong></a></p>
<p><strong>This library is currently under heavy development - if you have suggestions
on the API or use-cases you'd like to be covered, please open an github issue
or reach out. We'd love to hear about how you're using the library.</strong></p>
<p><code>functorch</code> is <a href="https://github.com/google/jax">JAX-like</a> composable function
transforms for PyTorch.</p>
<p>It aims to provide composable <code>vmap</code> and <code>grad</code> transforms that work with
PyTorch modules and PyTorch autograd with good eager-mode performance.</p>
<p>In addition, there is experimental functionality to trace through these
transformations using FX in order to capture the results of these transforms
ahead of time. This would allow us to compile the results of vmap or grad
to improve performance.</p>
<h2>Why composable function transforms?</h2>
<p>There are a number of use cases that are tricky to do in
PyTorch today:
- computing per-sample-gradients (or other per-sample quantities)
- running ensembles of models on a single machine
- efficiently batching together tasks in the inner-loop of MAML
- efficiently computing Jacobians and Hessians
- efficiently computing batched Jacobians and Hessians</p>
<p>Composing <code>vmap</code>, <code>grad</code>, <code>vjp</code>, and <code>jvp</code> transforms allows us to express the above
without designing a separate subsystem for each. This idea of composable function
transforms comes from the <a href="https://github.com/google/jax">JAX framework</a>.</p>
<h2>Install</h2>
<p>There are two ways to install functorch:
1. functorch from source
2. functorch beta (compatible with recent PyTorch releases)</p>
<p>We recommend trying out the functorch beta first.</p>
<h3>Installing functorch from source</h3>
<details><summary>Click to expand</summary>
<p>

#### Using Colab

Follow the instructions [in this Colab notebook](https://colab.research.google.com/drive/1CrLkqIrydBYP_svnF89UUO-aQEqNPE8x?usp=sharing)

#### Locally

As of 9/21/2022, `functorch` comes installed alongside a nightly PyTorch binary.
Please install a Preview (nightly) PyTorch binary; see  https://pytorch.org/
for instructions.

Once you've done that, run a quick sanity check in Python:
```py
import torch
from functorch import vmap
x = torch.randn(3)
y = vmap(torch.sin)(x)
assert torch.allclose(y, x.sin())
```

#### functorch development setup

As of 9/21/2022, `functorch` comes installed alongside PyTorch and is in the
PyTorch source tree. Please install
[PyTorch from source](https://github.com/pytorch/pytorch#from-source), then,
you will be able to `import functorch`.

Try to run some tests to make sure all is OK:
```bash
pytest test/test_vmap.py -v
pytest test/test_eager_transforms.py -v
```

AOTAutograd has some additional optional requirements. You can install them via:
```bash
pip install networkx
```

To run functorch tests, please install our test dependencies (`expecttest`, `pyyaml`).


</p>
</details>

<h3>Installing functorch beta (compatible with recent PyTorch releases)</h3>
<details><summary>Click to expand</summary>
<p>

#### Using Colab

Follow the instructions [here](https://colab.research.google.com/drive/1GNfb01W_xf8JRu78ZKoNnLqiwcrJrbYG#scrollTo=HJ1srOGeNCGA)

#### pip

Prerequisite: [Install PyTorch](https://pytorch.org/get-started/locally/)


```bash
pip install functorch
```

Finally, run a quick sanity check in python:
```py
import torch
from functorch import vmap
x = torch.randn(3)
y = vmap(torch.sin)(x)
assert torch.allclose(y, x.sin())
```

</p>
</details>

<h2>What are the transforms?</h2>
<p>Right now, we support the following transforms:
- <code>grad</code>, <code>vjp</code>, <code>jvp</code>,
- <code>jacrev</code>, <code>jacfwd</code>, <code>hessian</code>
- <code>vmap</code></p>
<p>Furthermore, we have some utilities for working with PyTorch modules.
- <code>make_functional(model)</code>
- <code>make_functional_with_buffers(model)</code></p>
<h3>vmap</h3>
<p>Note: <code>vmap</code> imposes restrictions on the code that it can be used on.
For more details, please read its docstring.</p>
<p><code>vmap(func)(*inputs)</code> is a transform that adds a dimension to all Tensor
operations in <code>func</code>. <code>vmap(func)</code> returns a new function that maps <code>func</code> over
some dimension (default: 0) of each Tensor in <code>inputs</code>.</p>
<p><code>vmap</code> is useful for hiding batch dimensions: one can write a function <code>func</code>
that runs on examples and then lift it to a function that can take batches of
examples with <code>vmap(func)</code>, leading to a simpler modeling experience:</p>
<p>```py
from functorch import vmap
batch_size, feature_size = 3, 5
weights = torch.randn(feature_size, requires_grad=True)</p>
<p>def model(feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()</p>
<p>examples = torch.randn(batch_size, feature_size)
result = vmap(model)(examples)
```</p>
<h3>grad</h3>
<p><code>grad(func)(*inputs)</code> assumes <code>func</code> returns a single-element Tensor. It compute
the gradients of the output of func w.r.t. to <code>inputs[0]</code>.</p>
<p>```py
from functorch import grad
x = torch.randn([])
cos_x = grad(lambda x: torch.sin(x))(x)
assert torch.allclose(cos_x, x.cos())</p>
<h1>Second-order gradients</h1>
<p>neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
assert torch.allclose(neg_sin_x, -x.sin())
```</p>
<p>When composed with <code>vmap</code>, <code>grad</code> can be used to compute per-sample-gradients:
```py
from functorch import vmap
batch_size, feature_size = 3, 5</p>
<p>def model(weights,feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()</p>
<p>def compute_loss(weights, example, target):
    y = model(weights, example)
    return ((y - target) ** 2).mean()  # MSELoss</p>
<p>weights = torch.randn(feature_size, requires_grad=True)
examples = torch.randn(batch_size, feature_size)
targets = torch.randn(batch_size)
inputs = (weights,examples, targets)
grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)
```</p>
<h3>vjp</h3>
<p>The <code>vjp</code> transform applies <code>func</code> to <code>inputs</code> and returns a new function that
computes vjps given some <code>cotangents</code> Tensors.
<code>py
from functorch import vjp
outputs, vjp_fn = vjp(func, inputs); vjps = vjp_fn(*cotangents)</code></p>
<h3>jvp</h3>
<p>The <code>jvp</code> transforms computes Jacobian-vector-products and is also known as
"forward-mode AD". It is not a higher-order function unlike most other transforms,
but it returns the outputs of <code>func(inputs)</code> as well as the <code>jvp</code>s.
<code>py
from functorch import jvp
x = torch.randn(5)
y = torch.randn(5)
f = lambda x, y: (x * y)
_, output = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
assert torch.allclose(output, x + y)</code></p>
<h3>jacrev, jacfwd, and hessian</h3>
<p>The <code>jacrev</code> transform returns a new function that takes in <code>x</code> and returns the
Jacobian of <code>torch.sin</code> with respect to <code>x</code> using reverse-mode AD.
<code>py
from functorch import jacrev
x = torch.randn(5)
jacobian = jacrev(torch.sin)(x)
expected = torch.diag(torch.cos(x))
assert torch.allclose(jacobian, expected)</code>
Use <code>jacrev</code> to compute the jacobian. This can be composed with vmap to produce
batched jacobians:</p>
<p><code>py
x = torch.randn(64, 5)
jacobian = vmap(jacrev(torch.sin))(x)
assert jacobian.shape == (64, 5, 5)</code></p>
<p><code>jacfwd</code> is a drop-in replacement for <code>jacrev</code> that computes Jacobians using
forward-mode AD:
<code>py
from functorch import jacfwd
x = torch.randn(5)
jacobian = jacfwd(torch.sin)(x)
expected = torch.diag(torch.cos(x))
assert torch.allclose(jacobian, expected)</code></p>
<p>Composing <code>jacrev</code> with itself or <code>jacfwd</code> can produce hessians:
```py
def f(x):
  return x.sin().sum()</p>
<p>x = torch.randn(5)
hessian0 = jacrev(jacrev(f))(x)
hessian1 = jacfwd(jacrev(f))(x)
```</p>
<p>The <code>hessian</code> is a convenience function that combines <code>jacfwd</code> and <code>jacrev</code>:
```py
from functorch import hessian</p>
<p>def f(x):
  return x.sin().sum()</p>
<p>x = torch.randn(5)
hess = hessian(f)(x)
```</p>
<h3>Tracing through the transformations</h3>
<p>We can also trace through these transformations in order to capture the results as new code using <code>make_fx</code>. There is also experimental integration with the NNC compiler (only works on CPU for now!).</p>
<p>```py
from functorch import make_fx, grad
def f(x):
    return torch.sin(x).sum()
x = torch.randn(100)
grad_f = make_fx(grad(f))(x)
print(grad_f.code)</p>
<p>def forward(self, x_1):
    sin = torch.ops.aten.sin(x_1)
    sum_1 = torch.ops.aten.sum(sin, None);  sin = None
    cos = torch.ops.aten.cos(x_1);  x_1 = None
    _tensor_constant0 = self._tensor_constant0
    mul = torch.ops.aten.mul(_tensor_constant0, cos);  _tensor_constant0 = cos = None
    return mul
```</p>
<h3>Working with NN modules: make_functional and friends</h3>
<p>Sometimes you may want to perform a transform with respect to the parameters
and/or buffers of an nn.Module. This can happen for example in:
- model ensembling, where all of your weights and buffers have an additional
dimension
- per-sample-gradient computation where you want to compute per-sample-grads
of the loss with respect to the model parameters</p>
<p>Our solution to this right now is an API that, given an nn.Module, creates a
stateless version of it that can be called like a function.</p>
<ul>
<li><code>make_functional(model)</code> returns a functional version of <code>model</code> and the
<code>model.parameters()</code></li>
<li><code>make_functional_with_buffers(model)</code> returns a functional version of
<code>model</code> and the <code>model.parameters()</code> and <code>model.buffers()</code>.</li>
</ul>
<p>Here's an example where we compute per-sample-gradients using an nn.Linear
layer:</p>
<p>```py
import torch
from functorch import make_functional, vmap, grad</p>
<p>model = torch.nn.Linear(3, 3)
data = torch.randn(64, 3)
targets = torch.randn(64, 3)</p>
<p>func_model, params = make_functional(model)</p>
<p>def compute_loss(params, data, targets):
    preds = func_model(params, data)
    return torch.mean((preds - targets) ** 2)</p>
<p>per_sample_grads = vmap(grad(compute_loss), (None, 0, 0))(params, data, targets)
```</p>
<p>If you're making an ensemble of models, you may find
<code>combine_state_for_ensemble</code> useful.</p>
<h2>Documentation</h2>
<p>For more documentation, see <a href="https://pytorch.org/functorch">our docs website</a>.</p>
<h2>Debugging</h2>
<p><code>torch._C._functorch.dump_tensor</code>: Dumps dispatch keys on stack
<code>torch._C._functorch._set_vmap_fallback_warning_enabled(False)</code> if the vmap warning spam bothers you.</p>
<h2>Future Plans</h2>
<p>In the end state, we'd like to upstream this into PyTorch once we iron out the
design details. To figure out the details, we need your help -- please send us
your use cases by starting a conversation in the issue tracker or trying our
project out.</p>
<h2>License</h2>
<p>Functorch has a BSD-style license, as found in the <a href="LICENSE">LICENSE</a> file.</p>
<h2>Citing functorch</h2>
<p>If you use functorch in your publication, please cite it by using the following BibTeX entry.</p>
<p><code>bibtex
@Misc{functorch2021,
  author =       {Horace He, Richard Zou},
  title =        {functorch: JAX-like composable function transforms for PyTorch},
  howpublished = {\url{https://github.com/pytorch/functorch}},
  year =         {2021}
}</code></p>