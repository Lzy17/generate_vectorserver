<h1>Data Sparsifier</h1>
<h2>Intro</h2>
<p>The data sparsifier inherits from the <code>BaseSparsifier</code> class. It attempts to sparsify data tensors in general (trainable and non-trainable).</p>
<h2>Implementation Details</h2>
<p>The data sparsifier does not receive a model or a layer to sparsify. Hence, the mask needs to be owned by the data sparsifier. This is achieved by introducing a private container model that registers the data as a parametrized buffer.</p>
<p>The BaseDataSparsifier handles all the housekeeping while allowing the user to just implement the <code>update_mask</code> logic in their implementation.</p>
<h2>Supported data</h2>
<ol>
<li>torch tensors (torch.Tensor)</li>
<li>parameters (nn.Parameter)</li>
<li>embedding and embedding bags (nn.Embeddings / nn.EmbeddingBag)</li>
</ol>
<h2>API details</h2>
<p><code>BaseDataSparsifier</code>: base class with abstract method <code>update_mask</code> that computes the new mask for all the data.</p>
<p><code>add_data</code>: Accepts name, data tuple and registers the data as a parametrized buffer inside the container model. Note that the data is always associated to a name. A custom sparse config can be provided along with the name, data pair. If not provided, the default config will be applied while doing the sparsification.
If the named data already exists, then it is replaced with the new data. The config and mask will be retained for the new data unless not specified to.
To not the old mask, set <code>reuse_mask=False</code>. If the <code>config</code> is explicitly passed in, it will be updated.</p>
<p><strong>Note</strong>: name containing '.' is not a valid name for the data sparsifier</p>
<p><code>data_sparsifier = ImplementedDataSparsifier()
data_sparsifier.add_data(name=name, data=data, **some_config)</code></p>
<p><code>step</code>: applies the update_mask() logic to all the data.</p>
<p><code>data_sparsifier.step()</code></p>
<p><code>get_mask</code>: retrieves the mask given the name of the data.</p>
<p><code>get_data</code>: retrieves the data given the <code>name</code> argument. Accepts additional argument <code>return_original</code> which when set to <code>True</code> does not apply the mask while returning
the data tensor. Example:</p>
<p><code>original_data = data_sparsifier.get_data(name=name, return_original=True)  # returns data with no mask applied
sparsified_data = data_sparsifier.get_data(name=name, return_original=False)  # returns data * mask</code></p>
<p><code>squash_mask</code>: removes the parametrizations on the data and applies mask to the data when <code>leave_parametrized=True</code>.Also, accepts list of strings to squash mask for. If none, squashes mask for all the keys.
<code>data_sparsifier.squash_mask()</code></p>
<p><code>state_dict</code>: Returns dictionary that can be serialized.</p>
<h2>Write your own data sparsifier.</h2>
<p>The custom data sparsifier should be inherited from the BaseDataSparsifier class and the <code>update_mask()</code> should be implemented. For example, the following data sparsifier zeros out all entries of the tensor smaller than some threshold value.</p>
<p>```
class ImplementedDataSparsifier(BaseDataSparsifier):
    def <strong>init</strong>(self, threshold):
        super().<strong>init</strong>(threshold=threshold)</p>
<pre><code>def update_mask(self, name, data, threshold):
    mask = self.get_mask(name)
    mask[torch.abs(data) &lt; threshold] = 0.0
</code></pre>
<p>```</p>
<h2>Using Data Sparsifier</h2>
<h3>Simple example</h3>
<p>```
tensor1 = torch.randn(100, 100)
param1 = nn.Parameter(torch.randn(200, 32))</p>
<p>my_sparsifier = ImplementedDataSparsifier(threshold=0.2)
my_sparsifier.add_data(name='tensor1', data=tensor1, threshold=0.5)
my_sparsifier.add_data(name='param1', data=param1)</p>
<p>my_sparsifier.step()  # computes mask</p>
<p>my_sparsifier.squash_mask()  # applies and removes mask
```</p>
<h3>Sparsifying model embeddings</h3>
<p>```
class Model(nn.Module):
    def <strong>init</strong>(self, feature_dim, emb_dim, num_classes):
        self.emb = nn.EmbeddingBag(feature_dim, emb_dim)
        self.linear1 = nn.Linear(emb_dim, 32)
        self.linear2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()</p>
<pre><code>def forward(self, x):
    out = self.emb(x)
    out = self.relu(self.linear1(out))
    out = self.linear2(out)
    return out
</code></pre>
<p>model = Model(100, 32, 10)
my_sparsifier = ImplementedDataSparsifier(threshold=0.5)
my_sparsifier.add_data(name='emb', data=model.emb)</p>
<p>...</p>
<h1>Train model</h1>
<p>...</p>
<p>my_sparsifier.step()  # creates mask for embeddings</p>
<p>my_sparsifier.squash_mask()  # applies and removes mask
```</p>
<h3>Using in the context of training data</h3>
<p>Sometimes if the input data can be sparsified before sending it to the model, then we can do so by using the data sparsifier.</p>
<p>The batched input data needs to be attached to the data sparsified before sending it to the model.</p>
<p>```
model = SomeModel()</p>
<p>data_sparsifier = ImplementedDataSparsifier(threshold=0.2)</p>
<p>data_name = 'train_data'</p>
<p>for x, y in train_data_loader:
    x = data_sparsifier.add_data(name=data_name, data=x)
    ...
    y_out = model(x)
    ...
    data_sparsifier.step()</p>
<p>```</p>
<p><strong>Note</strong>:
1. It is the responsibility of the <code>BaseDataSparsifier</code> to call the <code>self.update_mask</code> when appropriate.
2. The mask should be modified in place.</p>
<pre><code>Some valid inplace operations are:
1. Change a portion of a mask: `mask[:10] = torch.zeros(10)`
2. Use an inplace operator: `mask *= another_mask`
3. Change the underlying data: `mask.data = torch.zeros_like(mask)`

Non-inplace operations are not valid, and might lead to bugs. For example:

1. Reassignment of a mask: `mask = torch.zeros_like(mask)`
2. Non-inplace arithmetic operations: `mask = mask * another_mask`
</code></pre>
<ol>
<li>Data sparsifier <code>name</code> argument cannot have a '.' in it.</li>
</ol>