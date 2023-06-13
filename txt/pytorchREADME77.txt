<h1>Structured Pruning</h1>
<h2>Intro / Motivation</h2>
<p><strong>Pruning</strong> is the technique of removing parameters from a model to reduce the computational cost. The goal of pruning is to improve the performance of the model while maintaining it's accuracy.</p>
<h3>Unstructured vs. Structured Pruning</h3>
<p>One way to do this is to consider each parameter individually. This gives us the greatest granularity when pruning and is called <strong>unstructured pruning</strong>.</p>
<p>For example, consider a simple linear regression model that is parametrized by a weight tensor W.</p>
<p><code>W = [[1 2 3]
     [4 5 6]
     [7 1 9]]</code></p>
<p>We can prune the lowest absolute value elements in W in order to preserve as much information as possible.
Below we've removed three parameters from W.</p>
<p><code>W_pruned = [[0 0 3]
            [4 5 6]
            [7 0 9]]</code></p>
<p>Unfortunately, zeroing out parameters does not offer a speed-up to the model out of the box. We need custom sparse kernels that are designed to take advantage of sparsity to speed up computation. For more information about unstructured pruning check out our tutorials <a href="">here</a>.</p>
<p>However, if we zero out a row of parameters at a time instead of a single parameter, we can speed up computation by resizing the weight matrix. This is called <strong>structured pruning</strong> and is what this folder implements.</p>
<p>```
W_pruned = [[0 0 0] = [[4, 5, 6],
            [4 5 6]    [7, 1, 9]]
            [7 1 9]]</p>
<p>```</p>
<h3>Weight Resizing</h3>
<p>However, since the pruned weight tensor has a different shape than the original weight tensor, subsequent operations will cause an error due to this shape mismatch. We need to remove both the weights of the original weight tensor and the columns of subsequent tensors that correspond to the pruned rows.</p>
<p>You can see an example of this below for a model containing two linear layers, one parametrized by W and another by U</p>
<p><img alt="" src="./images/prune_5.png" /></p>
<p>By removing a row from U and a column from W, we can avoid a shape mismatch.</p>
<p><img alt="" src="./images/prune_6.png" /></p>
<p>One benefit of <strong>structured pruning</strong> is that it uses the same dense kernels that the original model uses, and does not rely on custom sparse kernel like <strong>unstructured pruning</strong>.
However, structured pruning degrades accuracy more than unstructured pruning because of the lack of granularity, so it is not always the right choice.</p>
<p>Generally the structured pruning process looks something like this:
1. Define what layers in the model you want to structured prune.
2. Evaluate the importance of each row in each layer in the model.
3. Remove rows by resizing the weight matrices of each layer
4. Stop if target sparsity level is met.</p>
<p>The accuracy degradation of pruning can be quite large initially. Once we are satisfied with our pruned tensor, we usually retrain the model after pruning in order to restore some of this accuracy loss.</p>
<h2>Quickstart Guide</h2>
<p><strong>Your model must be FX symbolically traceable</strong>.</p>
<p>You can test this with the following bit of code:</p>
<p><code>python
from torch.fx import symbolic_trace
model = MyModel()
symbolic_trace(model)</code></p>
<p>Using <code>torch.fx</code> we can get a compute graph of our model. Each operation (add, multiply, ReLU) is a node in the graph, and the order of operations is defined by the edges of the graph.</p>
<p>Structured pruning works by traversing this graph and looking for specific <strong>patterns</strong>, which are just a specific sequence of operations.</p>
<p>Each pattern is tied to a pruning function, which is responsible for structured pruning the graph nodes that match the pattern.</p>
<p>The above <a href="#weight-resizing">example</a> of two linear layers would match against a <code>(nn.Linear, nn.Linear)</code> pattern. This is how we identify the rows to remove and the columns of the subsequent layer.</p>
<p>Structured pruning also works on other patterns other than two adjacent Linear layers,</p>
<ul>
<li>linear -&gt; linear</li>
<li>linear -&gt; activation -&gt; linear</li>
<li>conv2d -&gt; conv2d</li>
<li>conv2d -&gt; activation -&gt; conv2d</li>
<li>conv2d -&gt; activation -&gt; pool -&gt; conv2d</li>
<li>conv2d -&gt; pool -&gt; activation -&gt; conv2d</li>
<li>conv2d -&gt; adaptive pool -&gt; flatten -&gt; linear</li>
</ul>
<p>A complete set of the patterns we support can be found <a href="https://github.com/pytorch/pytorch/blob/master/torch/ao/pruning/_experimental/pruner/base_structured_sparsifier.py#L85">here</a>.</p>
<p>If you are looking to prune a currently unsupported pattern, you can do this by modifying the pattern dict that we provide to the pruner, see <a href="#writing-custom-patterns-and-pruning-functions-for-structured-pruning">here</a>. Feel free to open a PR to add in new patterns.</p>
<p>Here is an example script that will prune away 50% of the rows for all the linear layers in the model, based on the saliency of each row.
```python
from torch.ao.pruning._experimental.pruner import SaliencyPruner</p>
<h1>Define model</h1>
<p>class Model(nn.Module):
    def <strong>init</strong>(self):
        super().<strong>init</strong>()
        self.seq = nn.Sequential(
            nn.Linear(700, 500, bias=True),
            nn.ReLU(),
            nn.Linear(500, 800, bias=False),
            nn.ReLU(),
            nn.Linear(800, 600, bias=True),
            nn.ReLU(),
        )
        self.linear = nn.Linear(600, 4, bias=False)</p>
<pre><code>def forward(self, x):
    x = self.seq(x)
    x = self.linear(x)
    return x
</code></pre>
<h1>Define pruning_config, which specifies which tensors you wish to prune.</h1>
<h1>The SaliencyPruner also needs a sparsity_level parameter to specify what % of rows to prune.</h1>
<p>pruning_config = [
    {"tensor_fqn": "seq.0.weight", "sparsity_level": 0.5},
    {"tensor_fqn": "seq.2.weight", "sparsity_level": 0.5},
    {"tensor_fqn": "seq.4.weight", "sparsity_level": 0.5},
    {"tensor_fqn": "linear.weight", "sparsity_level": 0.5},
]</p>
<p>original = Model()</p>
<h1>define defaults</h1>
<h1>for structured pruning, we also prune biases by default.</h1>
<p>defaults = {"prune_bias": True}</p>
<h1>any configs passed in here are defaults that are propagated</h1>
<h1>Your selection criteria is decided by which pruner you use</h1>
<p>pruner = SaliencyPruner(defaults, patterns=patterns)</p>
<h1>Next we call <code>prepare</code>, which will attach <code>FakeStructuredSparsity</code> parameterizations</h1>
<h1>to the tensors specified in the config. These parameterizations will zero out</h1>
<h1>the appropriate weights in order to make the model behave as if it has been pruned.</h1>
<p>pruner.prepare(original, sparse_config)</p>
<h1>take one pruning step. This will update the masks</h1>
<p>pruner.enable_mask_update = True
pruner.step()</p>
<h1>pruner.prune() will find patterns and apply that patterns pruning function to it's matching nodes.</h1>
<h1>The output of pruner.prune() is a model with resized weights and the masks / parametrizations removed.</h1>
<p>pruned_model = pruner.prune()
```
Afterwards, by printing the name and size of each parameter in our model, we can see that it has been pruned.</p>
<p>```</p>
<h1>original model</h1>
<p>Parameter name      | Shape           |  # of elements
--------------------|-----------------|---------------
seq.0.weight        | 500, 700        |    350000
seq.0.bias          | 500             |       500
seq.2.weight        | 800, 500        |    400000
seq.4.weight        | 600, 800        |    480000
seq.4.bias          | 600             |       600
linear.weight       | 4, 600          |      2400
=== Total Number of Parameters: 1233500 ===
<code></code></p>
<h1>pruned model</h1>
<p>Parameter name      | Shape           |  # of elements
--------------------|-----------------|---------------
seq.0.weight        | 250, 700        |    175000
seq.0.bias          | 250             |       250
seq.2.weight        | 400, 250        |    100000
seq.4.weight        | 300, 400        |    120000
seq.4.bias          | 300             |       300
linear.weight       | 2, 300          |       600
=== Total Number of Parameters: 396150 ===
```</p>
<p>Although we pruned 50% of the rows, the total number of parameters is 25% of the original model.</p>
<p>Since we remove both the rows of a weight tensor and the columns of the subsequent tensor. The total number of parameters is roughly (1-0.5)* (1-0.5) = 0.25 of the original number of parameters.</p>
<h2>Advanced Tutorial</h2>
<h3>Pruning Config</h3>
<p>To specify the layers to prune we just need the fully qualified name (FQN) of the tensor you are looking to prune in the module.
You can get the FQN of a tensor by printing out <code>model.named_parameters()</code>.</p>
<p>To prune multiple layers, we just append entries to the pruning config.
<strong>tensor_fqn</strong> is the only required key in the pruning config. You can pass additional information in the config, for example the sparsity level you want to prune to by adding a key to the config. You can then access this additional information when you update the masks.</p>
<h3>Implementing a Pruner</h3>
<p>If you want to prune weights using a different pruning criteria than saliency, you'll need to implement your own pruner.</p>
<p>To do this, we need to extend a <code>BaseStructuredSparsifier</code> with a custom <code>update_mask</code> function.</p>
<p>This <code>update_mask</code> function contains the user logic for picking what weights to prune.</p>
<p>One common pruning criteria is to use the <strong>saliency</strong> of a row, which is defined as the sum of all the L1 norms of the weights in the row.
The idea is to remove the weights that are small, since they wouldn't contribute much to the final prediction.</p>
<p>Below we can see an implemented Saliency Pruner</p>
<p>```python
class SaliencyPruner(BaseStructuredSparsifier):
     """
     Prune filters based on the saliency
     The saliency for a filter is given by the sum of the L1 norms of all of its weights
     """</p>
<pre><code> def update_mask(self, module, tensor_name, **kwargs):
    # tensor_name will give you the FQN, all other keys in pruning config are present in kwargs
     weights = getattr(module, tensor_name)
     mask = getattr(module.parametrizations, tensor_name)[0].mask

     # use negative weights so we can use topk (we prune out the smallest)
     saliency = -weights.norm(dim=tuple(range(1, weights.dim())), p=1)
     num_to_pick = int(len(mask) * kwargs["sparsity_level"])
     prune = saliency.topk(num_to_pick).indices

     # Set the mask to be false for the rows we want to prune
     mask.data[prune] = False
</code></pre>
<p>```</p>
<h3>Writing Custom Patterns and Pruning Functions for Structured Pruning</h3>
<p>If you're working with linear/conv2d layers, it's very probable that you just need to add an entry to the pattern dict mapping your pattern to an existing prune_function.</p>
<p>This is because there are many modules, for example <strong>pooling</strong> that behave the same way and do not need to be modified by the pruning code.</p>
<p>```python
from torch.ao.pruning._experimental.pruner.prune_functions import prune_conv2d_activation_conv2d</p>
<p>def prune_conv2d_pool_activation_conv2d(
    c1: nn.Conv2d,
    pool: nn.Module,
    activation: Optional[Callable[[Tensor], Tensor]],
    c2: nn.Conv2d,
) -&gt; None:
    prune_conv2d_activation_conv2d(c1, activation, c2)</p>
<h1>note how the pattern defined in the key will be passed to the pruning function as args</h1>
<p>my_patterns = {(nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Conv2d): prune_conv2d_activation_conv2d}</p>
<p>pruning_patterns = _get_default_structured_pruning_patterns()
pruning_patterns.update(my_patterns)</p>
<p>pruner = SaliencyPruner({}, patterns=pruning_patterns)
```
However, there are also modules like batch norm, which will not work properly without being pruned as well. In this instance, you would need to write a custom pruning function in order to handle that logic properly.</p>
<p>You can see the implemented pruning functions <a href="https://github.com/pytorch/pytorch/blob/master/torch/ao/pruning/_experimental/pruner/prune_functions.py">here</a> for examples. Please feel free to open a PR so we get a complete set of the patterns and pruning functions.</p>