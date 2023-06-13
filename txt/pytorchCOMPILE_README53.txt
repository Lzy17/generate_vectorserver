<h1>AOT Autograd - Introduction to an experimental compilation feature in Functorch</h1>
<p>The primary compilation API we provide is something called AOTAutograd. AOT
Autograd is an experimental feature that allows ahead of time capture of forward
and backward graphs, and allows easy integration with compilers. This creates an
easy to hack Python-based development environment to speedup training of PyTorch
models. AOT Autograd currently lives inside functorch.compile namespace.</p>
<p>AOT Autograd is experimental and the APIs are likely to change. We are looking
for feedback. If you are interested in using AOT Autograd and need help or have
suggestions, please feel free to open an issue. We will be happy to help.</p>
<p>For example, here are some examples of how to use it.
```python
from functorch.compile import aot_function, aot_module, draw_graph
import torch.fx as fx
import torch</p>
<h1>This simply prints out the FX graph of the forwards and the backwards</h1>
<p>def print_graph(name):
    def f(fx_g: fx.GraphModule, inps):
        print(name)
        print(fx_g.code)
        return fx_g
    return f</p>
<p>def f(x):
    return x.cos().cos()</p>
<p>nf = aot_function(f, fw_compiler=print_graph("forward"), bw_compiler=print_graph("backward"))
nf(torch.randn(3, requires_grad=True))</p>
<h1>You can do whatever you want before and after, and you can still backprop through the function.</h1>
<p>inp = torch.randn(3, requires_grad=True)
inp = inp.cos()
out = nf(inp)
out = out.sin().sum().backward()</p>
<p>def f(x):
    return x.cos().cos()</p>
<h1>This draws out the forwards and the backwards graphs as svg files</h1>
<p>def graph_drawer(name):
    def f(fx_g: fx.GraphModule, inps):
        draw_graph(fx_g, name)
        return fx_g
    return f</p>
<p>aot_function(f, fw_compiler=graph_drawer("forward"), bw_compiler=graph_drawer("backward"))(torch.randn(3, requires_grad=True))</p>
<h1>We also have a convenience API for applying AOTAutograd to modules</h1>
<p>from torchvision.models import resnet18
aot_module(resnet18(), print_graph("forward"), print_graph("backward"))(torch.randn(1,3,200,200))</p>
<h1>output elided since it's very long</h1>
<h1>In practice, you might want to speed it up by sending it to Torchscript. You might also lower it to Torchscript before passing it to another compiler</h1>
<p>def f(x):
    return x.cos().cos()</p>
<p>def ts_compiler(fx_g: fx.GraphModule, inps):
    f = torch.jit.script(fx_g)
    print(f.graph)
    f = torch.jit.freeze(f.eval()) # Note: This eval() works fine <em>even</em> though we're using this for training
    return f</p>
<p>aot_function(f, ts_compiler, ts_compiler)(torch.randn(3, requires_grad=True))
```</p>
<h2>Documentation</h2>
<ul>
<li>AOT Autograd <a href="https://pytorch.org/functorch/nightly/">documentation</a></li>
<li>Min-cut <a href="https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467">recomputation</a> with AOT Autograd.</li>
</ul>
<h2>Tutorials</h2>
<p>You can use this <a href="https://pytorch.org/functorch/nightly/notebooks/aot_autograd_optimizations.html">tutorial</a> to play with AOT Autograd.</p>