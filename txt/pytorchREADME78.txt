<h1>Activation Sparsifier</h1>
<h2>Introduction</h2>
<p>Activation sparsifier attaches itself to a layer(s) in the model and prunes the activations passing through them. <strong>Note that the layer weights are not pruned here.</strong></p>
<h2>How does it work?</h2>
<p>The idea is to compute a mask to prune the activations. To compute the mask, we need a representative tensor that generalizes activations coming from all the batches in the dataset.</p>
<p>There are 3 main steps involved:
1. <strong>Aggregation</strong>: The activations coming from inputs across all the batches are aggregated using a user-defined <code>aggregate_fn</code>.
A simple example is the add function.
2. <strong>Reduce</strong>: The aggregated activations are then reduced using a user-defined <code>reduce_fn</code>. A simple example is average.
3. <strong>Masking</strong>: The reduced activations are then passed into a user-defined <code>mask_fn</code> to compute the mask.</p>
<p>Essentially, the high level idea of computing the mask is</p>
<p>```</p>
<blockquote>
<blockquote>
<blockquote>
<p>aggregated_tensor = aggregate_fn([activation for activation in all_activations])
reduced_tensor = reduce_fn(aggregated_tensor)
mask = mask_fn(reduced_tensor)
```</p>
</blockquote>
</blockquote>
</blockquote>
<p><em>The activation sparsifier also supports per-feature/channel sparsity. This means that a desired set of features in an activation can be also pruned. The mask will be stored per feature.</em></p>
<p>```</p>
<blockquote>
<blockquote>
<blockquote>
<h1>when features = None, mask is a tensor computed on the entire activation tensor</h1>
<h1>otherwise, mask is a list of tensors of length = len(features), computed on each feature of activations</h1>
<h1>On a high level, this is how the mask is computed if features is not None</h1>
<p>for i in range(len(features)):
   aggregated_tensor_feature = aggregate_fn([activation[features[i]] for activation in all_activations])
   mask[i] = mask_fn(reduce_fn(aggregated_tensor_feature))
```</p>
</blockquote>
</blockquote>
</blockquote>
<h2>Implementation Details</h2>
<p>The activation sparsifier attaches itself to a set of layers in a model and then attempts to sparsify the activations flowing through them. <em>Attach</em> means registering a <a href="https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#register_forward_pre_hook"><code>forward_pre_hook()</code></a> to the layer.</p>
<p>Let's go over the 3 steps again -
1. <strong>Aggregation</strong>: The activation of aggregation happens by attaching a hook to the layer that specifically applies and stores the aggregated data. The aggregation happens per feature, if the features are specified, otherwise it happens on the entire tensor.
The <code>aggregate_fn</code> should accept two input tensors and return an aggregated tensor. Example:
<code>def aggregate_fn(tensor1, tensor2):
    return tensor1 + tensor2</code></p>
<ol>
<li>
<p><strong>Reduce</strong>: This is initiated once the <code>step()</code> is called. The <code>reduce_fn()</code> is called on the aggregated tensor. The goal is to squash the aggregated tensor.
The <code>reduce_fn</code> should accept one tensor as argument and return a reduced tensor. Example:
<code>def reduce_fn(agg_tensor):
    return agg_tensor.mean(dim=0)</code></p>
</li>
<li>
<p><strong>Masking</strong>: The computation of the mask happens immediately after the reduce operation. The <code>mask_fn()</code> is applied on the reduced tensor. Again, this happens per-feature, if the features are specified.
The <code>mask_fn</code> should accept a tensor (reduced) and sparse config as arguments and return a mask (computed using tensor according to the config). Example:
<code>def mask_fn(tensor, threshold):  # threshold is the sparse config here
    mask = torch.ones_like(tensor)
    mask[torch.abs(tensor) &lt; threshold] = 0.0
    return mask</code></p>
</li>
</ol>
<h2>API Design</h2>
<p><code>ActivationSparsifier</code>: Attaches itself to a model layer and sparsifies the activation flowing through that layer. The user can pass in the default <code>aggregate_fn</code>, <code>reduce_fn</code> and <code>mask_fn</code>. Additionally, <code>features</code> and <code>feature_dim</code> are also accepted.</p>
<p><code>register_layer</code>: Registers a layer for sparsification. Specifically, registers <code>forward_pre_hook()</code> that performs aggregation.</p>
<p><code>step</code>: For each registered layer, applies the <code>reduce_fn</code> on aggregated activations and then applies <code>mask_fn</code> after reduce operation.</p>
<p><code>squash_mask</code>: Unregisters aggregate hook that was applied earlier and registers sparsification hooks if <code>attach_sparsify_hook=True</code>. Sparsification hooks applies the computed mask to the activations before it flows into the registered layer.</p>
<h2>Example</h2>
<p>```</p>
<h1>Fetch model</h1>
<p>model = SomeModel()</p>
<h1>define some aggregate, reduce and mask functions</h1>
<p>def aggregate_fn(tensor1, tensor2):
    return tensor1 + tensor2</p>
<p>def reduce_fn(tensor):
    return tensor.mean(dim=0)</p>
<p>def mask_fn(data, threshold):
    mask = torch.ones_like(tensor)
    mask[torch.abs(tensor) &lt; threshold] = 0.0
    return mask)</p>
<h1>sparse config</h1>
<p>default_sparse_config = {"threshold": 0.5}</p>
<h1>define activation sparsifier</h1>
<p>act_sparsifier = ActivationSparsifier(model=model, aggregate_fn=aggregate_fn, reduce_fn=reduce_fn, mask_fn=mask_fn, **threshold)</p>
<h1>register some layer to sparsify their activations</h1>
<p>act_sparsifier.register_layer(model.some_layer, threshold=0.8)  # custom sparse config</p>
<p>for epoch in range(EPOCHS):
    for input, target in dataset:
        ...
        out = model(input)
        ...
    act_sparsifier.step()  # mask is computed</p>
<p>act_sparsifier.squash_mask(attach_sparsify_hook=True)  # activations are multiplied with the computed mask before flowing through the layer
```</p>