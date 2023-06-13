<h1>Named Tensors using First-class Dimensions in PyTorch</h1>
<p>-- Zachary DeVito <a href="https://twitter.com/Zachary_DeVito">@Zachary_DeVito</a></p>
<p><em>An implementation of <a href="https://namedtensor.github.io">named tensors</a> with the functionality of <a href="http://einops.rocks]http://einops.rocks">einsum</a> , batching (<a href="https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap">vmap</a>, <a href="https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html">xmap</a>), and tensor indexing by adding dimension objects to PyTorch</em>.</p>
<p>The tensor input to a resnet might have the shape [8, 3, 224, 224] but informally we think of those dimensions as 'batch', 'channel', 'width', and 'height'. Eventhough 'width' and 'height' have the same <em>size</em> we still think of them as separate dimensions, and if we have two <em>different</em> images, we think of both as sharing the <em>same</em> 'channel' dimension.</p>
<p>Named tensors gives these dimensions names. <a href="https://pytorch.org/docs/stable/named_tensor.html">PyTorch's current implementation</a> uses strings to name dimensions. Instead, this library introduces a Python object, a <code>Dim</code>, to represent the concept. By expanding the semantics of tensors with dim objects, in addition to naming dimensions, we can get behavior equivalent to batching transforms (xmap, vmap), einops-style rearrangement, and loop-style tensor indexing.</p>
<p>A preview:</p>
<p>```py
from torchdim import dims</p>
<h1>einsum</h1>
<p>def mm(A: torch.Tensor, B: torch.Tensor):
    i, j, k = dims(3)
    r = (A[i, k] * B[k, j]).sum(k)
    return r.order(i, j)</p>
<h1>rearrange</h1>
<p>def pixel_shuffle(img: torch.Tensor, upscale_factor=2):
    h2, w2, c, b, h, w = dims(6)
    h2.size = w2.size = upscale_factor
    return img[b, (c, h2, w2), h, w].order(b, c, (h, h2), (w, w2))</p>
<h1>batching</h1>
<p>def bmm(A: torch.Tensor, B: torch.Tensor):
    i = dims(1)
    return mm(A[i], B[i]).order(i)</p>
<h1>indexing</h1>
<p>def embedding_bag(input: torch.Tensor, embedding_weights: torch.Tensor):
    batch, sequence, features = dims(3)
    r = embedding_weights[input[batch, sequence], features].sum(sequence)
    return r.order(batch, features)
```</p>
<h1>Installation</h1>
<p><em>torchdim is a preview release so that we can collect feedback on the API. It may have bugs, and there are known places where performance can be improved.</em></p>
<p>First-class dims are a library that extends PyTorch, so they need to be installed separately.
We may eventually upstream them into PyTorch itself along with <code>functorch</code>.</p>
<p>We have to install a nightly build of PyTorch so first set up an environment:</p>
<p><code>sh
conda create --name dim
conda activate dim</code></p>
<p>First-class dims requires a fairly recent nightly build of PyTorch so that functorch will work. You can install it using one of these commands:</p>
<p>```sh</p>
<h1>For CUDA 10.2</h1>
<p>conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-nightly</p>
<h1>For CUDA 11.3</h1>
<p>conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly</p>
<h1>For CPU-only build</h1>
<p>conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly
```</p>
<p>Install dim. You will be asked for github credentials to access the fairinternal organization.</p>
<p><code>sh
pip install ninja  # Makes the build go faster
pip install --user "git+https://github.com/facebookresearch/torchdim"</code></p>
<h1>Creating and Binding Dims</h1>
<p>Python objects that represent dimension are created using the <code>dims</code> operator.[^1]</p>
<p>```py
import torch
from torchdim import dims</p>
<p>batch, channel, width, height = dims(4)
```</p>
<p>The existing implementation of <a href="https://pytorch.org/docs/stable/named_tensor.html">Named Tensors</a> in PyTorch, or <a href="https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html">JAX's xmap</a> use strings to name dimensions. We call these dimensions <em>first class</em> because they are Python objects.</p>
<p>In addition to the normal <em>positional</em> dimensions in a tensor, tensors can also have a separate set of first-class dimensions.</p>
<p>You can create tensors with first-class dimensions by indexing the normal positional dimensions of a tensor with a dimension object. The <code>ndim</code> property continues to list the number of positional dimensions, while the new <code>dims</code> property lists all the bound first-class dimensions.</p>
<p>```py
input = torch.rand(2, 3, 224, 224)
print(input.ndim)</p>
<blockquote>
<p>4</p>
</blockquote>
<p>input_fc = input[batch, channel, width, height]
print(input_fc.dims) # first class dimensions</p>
<blockquote>
<p>(batch, channel, width, height)</p>
</blockquote>
<h1>since we converted all the positional dimensions</h1>
<h1>first class <code>input_fc</code> has 0 positional dimensions now.</h1>
<p>print(input_fc.ndim)</p>
<blockquote>
<p>0
```</p>
</blockquote>
<p>Notice that indexing creates a <em>new</em> Tensor, <code>input_fc</code> with bound first-class dimensions. It does not modify the original tensor <code>input</code>, which still has 4 positional dimensions.</p>
<p>```py
print(input.ndim) # unchanged</p>
<blockquote>
<p>4
```</p>
</blockquote>
<p>Importantly, indexing with square brackets <em>applies only to positional dimensions</em>, so attempting to index a tensor with only first class dims will error[^2]:</p>
<p>```py
try:
    input_fc[0]
except ValueError as ve:
    print(ve)</p>
<blockquote>
<p>at least 1 indices were supplied but the tensor only has 0 dimensions
```</p>
</blockquote>
<p>Generally, it is possible to construct tensors with a mixture of positional and first class dimensions:</p>
<p>```py
input_mixed = input[batch, :, :, height]
print(input_mixed.dims)</p>
<blockquote>
<p>(batch, height)</p>
</blockquote>
<p>print(input_mixed.ndim)</p>
<blockquote>
<p>2
```</p>
</blockquote>
<h2>Dimension Sizes</h2>
<p>Dimensions will take on the size of the first thing they are bound to:</p>
<p>```py
input = torch.rand(3)
x = dims(1)
input_fc = input[x]
print(x.size)</p>
<blockquote>
<p>3
```</p>
</blockquote>
<p>But you can also directly set the size of dimension:</p>
<p>```py
i = dims(1)</p>
<p>i.size = 5 # ok, i previously did not have a size</p>
<p>i.size = 5 # ok, it already had the size 5
try:
    i.size = 3
except Exception as e:
    print(e)</p>
<blockquote>
<p>Dim 'i' previously bound to a dimension of size 5 cannot bind to a dimension of size 3</p>
</blockquote>
<p>j = dims(sizes=[4]) # can also be set on construction
```</p>
<p>[^1]: We use a bit of Python introspection to set the debug names for the dimensions based on the names of the variables they are assigned to.
[^2]: Indexing of first-class dimensions can be done with the <code>index</code> method by specifying the dimension to be index into (e.g. <code>input_fc.index(batch, 0)</code>.</p>
<h1>Semantics of Dimensions</h1>
<p>The power of named tensors arises from how the first-class dimensions in the Tensors composed with existing operations.</p>
<p>Three rules define how dimension objects behave with existing Tensors.</p>
<h2>Rule 1: Implicit Batching</h2>
<p><strong>Tensor operations (e.g. <code>input + bias</code>) are implicitly batched over the union of the first-class dimensions in their inputs.</strong></p>
<p>If <code>input</code> has dimensions <code>batch, channel</code> and <code>bias</code> has dimension <code>channel</code>, the output will have the union of those dimensions (<code>batch, channel</code>), and the result will computed as if there was a loop over all the first-class dimensions.[^3]</p>
<p>```py
input_positional = torch.rand(128, 32)
bias_positional = torch.rand(32)</p>
<p>batch, channel = dims(2)
input = input_positional[batch, channel]
bias = bias_positional[channel]</p>
<p>result = input + bias
print(result.dims)</p>
<blockquote>
<p>(batch, channel)
```</p>
</blockquote>
<p>It is helpful think of operators on tensors with first-class dimensions by analogy to code with explicit loops over dimensions, with the first-class dimensions of the inputs acting as implicit <code>for</code> loops, and the values in the tensor being scalars within the body of the loop:</p>
<p>```py</p>
<h1>mental model: loop-level analogy</h1>
<p>for batch in range(batch.size):
    for channel in range(channel.size):
        input = input_positional[batch, channels]
        bias = bias_positional[channels]
        result[batch, channels] =  input + bias # arithmetic on scalars
```</p>
<p>Positional dimensions behave as they did before (e.g. for + they will broadcast), and can be thought of as being a standard tensor <em>used within the implicit loops</em> defined by first-class dimensions.</p>
<p>In this example, we broke down the expression into lines that bind the dimension to positional tensors and then another line to do the compute. In practice, we often combine these in one statement:</p>
<p><code>py
result = input_positional[batch, channel] + bias_positional[channel]
result.dims</code></p>
<p>[^3] This rule is similar to how named dimensions in xmap behave within a function, but instead of introducing the dimensions via a functional transform, they are bound on the objects using indexing.</p>
<h2>Rule 2: Specifying dimensions</h2>
<p><strong>Wherever an integer is used to specify a dimension in the existing torch operator, a first-class dimensions can be used instead to tell the operator to work over that dimension.</strong></p>
<p>```py
batch, channel, width, height = dims(4)
input_positional = torch.rand(2, 3, 224, 224)
input = input_positional[batch, channel, width, height]
avg_pixel_color = input.mean((width, height))</p>
<p>print(avg_pixel_color.dims)</p>
<blockquote>
<p>(batch, channel)
```</p>
</blockquote>
<p>Any other first-class dimensions (e.g. batch, channel) are still implicitly batched according to Rule #1.</p>
<h2>Rule 3: Dims are Tensors</h2>
<p><strong>A first-class dimension <code>d</code> can be used wherever a Tensor is expected. It will act as if it were a tensor whose only dimension is itself, <code>d</code>, and the values along the dimension are the indices of each entry <code>(0, 1, 2, ..., d.size - 1)</code></strong></p>
<p>```py
print(channel.dims)</p>
<blockquote>
<p>(channel,)</p>
</blockquote>
<p>print(channel + 1000)</p>
<blockquote>
<p>tensor([1000, 1001, 1002])
with dims=(channel,) sizes=(3,)
```</p>
</blockquote>
<p>This means that a dimension used as a tensor acts as an index into that dimension. Going back to our loop-level analogy, it is analogous to using the loop variable as a value:</p>
<p>```py</p>
<h1>mental model: loop-level analogy</h1>
<p>for channel in range(batch.size):
    result[channel] = channel + 1000
```</p>
<p>Arithmetic using dimension indices comes up a lot, such as the mask for an upper triangular part of a matrix. Using dims as tensors makes it easy:</p>
<p>```py
from torchdim import dims
i, j = dims(sizes=[4, 4])
print(i &lt;= j)</p>
<blockquote>
<p>tensor([[ True,  True,  True,  True],
        [False,  True,  True,  True],
        [False, False,  True,  True],
        [False, False, False,  True]])
with dims=(i, j) sizes=(4, 4)
```</p>
</blockquote>
<p>Because of the intentional similarity to loop-level code, using dimensions as tensors makes complicated indexing arithmetic easier to read.</p>
<p>Here is code that lookups up features in an embedding table given a sequence of ids:</p>
<p>```py
sequence, features = dims(2)
embeddings = torch.rand(8, 128)
words = torch.tensor([5, 4, 0,])</p>
<p>state = embeddings[words[sequence], features]
print(state.dims)</p>
<blockquote>
<p>(sequence, features)
```</p>
</blockquote>
<p>With the following analogy to loops:</p>
<p>```py</p>
<h1>mental model: loop-level analogy</h1>
<p>for sequence in range(words.size(0)):
    for features in range(embeddings.size(1)):
        state = embeddings[words[sequence], features]
```</p>
<p>Earlier we showed how binding tensors dimension is done with indexing <code>A[i, j]</code>. In fact, this binding is just the normal indexing operator. Its behavior follows directly from the behavior of indexing with tensor indices combined with Rule #3 and Rule #1. The expression <code>A[i + 1, j]</code> also creates a tensor with dimensions <code>i</code> and <code>j</code> but with different indexing math. The implementation knows when simple indexing patterns are used and only actually runs a kernel to do indexing when needed.</p>
<h2>Unbinding Dims</h2>
<p>The <code>order</code> method converts first-class dimensions in a tensor back to normal positional dimensions by specifying an order for those dimensions.[^4]</p>
<p>By specifying a different order from how things were originally bound, it is easy to do transpositions.</p>
<p><code>py
i, j = dims(2)
A = torch.rand(3, 4)
A_T = A[i, j].order(j, i)
assert torch.allclose(A.T, A_T)</code></p>
<p>Indexing acts left-to-right, and <code>order</code> also places the new dimensions back on the left, so it possible to work on tensors that have mixed positional and first-class dimensions:</p>
<p><code>py
B = torch.rand(3, 4, 5)
B_T = B[i, j].order(j, i)
assert torch.allclose(B.permute(1, 0, 2), B_T)</code></p>
<p>[^4] <code>order</code> is actually just a synonym for the already-existing <code>permute</code> method, which takes a list a dimension specifiers and puts the tensor in that order because rule #2 says that first-class dims can be passed as arguments to functions that previously took only integers as dimensions. However, the name <code>permute</code> is confusing in this context since it implies dim objects have an original order, so we prefer to use <code>order</code> when writing code.</p>
<h2>Flattening and Splitting Dims</h2>
<p><strong>Tuples of dimensions</strong> can be passed to both indexing and <code>order</code>. In indexing, this will split the dimension being indexed across the dimensions in the tuple.  In <code>order</code> it will flatten the dimensions in a single positional dimension:</p>
<p>```py
i, j, k = dims(3)
j.size = 2
A = torch.rand(6, 4)
a = A[(i, j), k] # split dim 0 into i,j
print(i.size, j.size, k.size)</p>
<blockquote>
<p>3 2 4</p>
</blockquote>
<p>r = a.order(i, (j, k)) # flatten j and k
print(r.shape)</p>
<blockquote>
<p>torch.Size([3, 8])
```</p>
</blockquote>
<p>The size of one unsized dimension in a tuple such as <code>i</code> can be inferred if the other sizes are known.</p>
<h1>Examples</h1>
<p>The usefulness of dimension objects is best seen through examples. Let's look at some different ways they can be used.</p>
<h2>Einsum-style Products</h2>
<p>Rather than having <a href="https://pytorch.org/docs/stable/generated/torch.einsum.html">einsum</a> as a custom operator, it is possible to express matrix products directly as a composition of multiplies and summations. The implementation will pattern match any multiplication followed by a sum to the right matrix-multiply operator.</p>
<p><code>py
def mm(A, B):
    i, j, k = dims(3)
    r = (A[i, k] * B[k, j]).sum(k)
    return r.order(i, j)
mm(torch.rand(3, 4), torch.rand(4, 5)).shape</code></p>
<p>The implementation of named tensors delays the execution of multiply to see if a summation follows it as it does above. If so, it will turn this pattern into the correct <em>optimized matrix product</em>, similar to how the <code>einsum</code> function works.</p>
<p>Since it is no longer necessary to manually match math to matrix functions, other tensor products are easier to express, like the Gram matrix used in style transfer:</p>
<p>```py
def gram_matrix_new(y):
    b, c, c2, h, w = dims()
    r = (y[b, c, h, w] * y[b, c2, h, w]).sum((h, w))
    r = r / (h.size * w.size)
    return r.order(b, c, c2)</p>
<p>gram_matrix_new(torch.rand(1, 2, 3, 4))</p>
<h1>[example adapted from http://einops.rocks/pytorch-examples.html]</h1>
<p>```</p>
<p>Attention is another example that has several matrix products embedded inside it:</p>
<p>```py
from torchdim import softmax
def attention(K, Q, V):
    batch, channel, key, query = dims(4)
    k = K[batch, channel, key]
    q = Q[batch, channel, query]
    v = V[batch, channel, key]</p>
<pre><code>a = (k * q).sum(channel) # matrix multiply
a = softmax(a * (channel.size ** -0.5), dim=key)
r = (v * a).sum(key) # matrix multiply
return torch.cat((r.order(batch, channel, query), Q), dim=1)
</code></pre>
<p>inputs = (torch.rand(2, 3, 4) for _ in range(3))
attention(*inputs)</p>
<h1>[example adapted from http://einops.rocks/pytorch-examples.html]</h1>
<p>```</p>
<h2>Reshaping tensors (einops)</h2>
<p>Lots of operations in deep learning are just different ways of reshaping, splitting, and joining dimensions, such as the pixel shuffle used to upscale an image by turning channels into pixels:</p>
<p><code>py
def pixel_shuffle(img, upscale_factor=2):
    h2, w2, c, b, h, w = dims(6)
    h2.size = w2.size = upscale_factor
    return img[b, (c, h2, w2), h, w].order(b, c, (h, h2), (w, w2))</code></p>
<p><a href="http://einops.rocks">Einops</a> is an extension to einsum that adds support for the manipulation of dimensions through a few custom operators such as <code>rearrange</code>:</p>
<p><code>py
def pixel_shuffle_einops(img, upscale_factor=2):
    from einops import rearrange
    return rearrange(img, 'b (c h2 w2) h w -&gt; b c (h h2) (w w2)', h2=upscale_factor, w2=upscale_factor)</code></p>
<p>Named tensors with first-class dimensions can accomplish the same goal, but using PyTorch's existing operator set.</p>
<h2>Automatically batching Code (<code>vmap</code>, <code>xmap</code>)</h2>
<p>The implicit batching of Rule #1 means it is easy to created batched versions of existing PyTorch code. Simply bind a dim to the dimensions that should act as a batch, and then pass the tensor to the unbatched function. Since the unbatched function does not know about the dim, the dim will be implicitly batched over:</p>
<p>```py
batch_size, feature_size = 3, 5
weights = torch.randn(feature_size)</p>
<p>def model(feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()</p>
<p>examples = torch.randn(batch_size, feature_size)
batch = dims(1)
r = model(examples[batch])
print(r)</p>
<h1>in functorch: result = functorch.vmap(model)(examples)</h1>
<blockquote>
<p>tensor([0.4775, 0.0000, 0.3423])
with dims=(batch,) sizes=(3,)
```</p>
</blockquote>
<p>This pattern also composes well with other code that also uses first class dimensions. For instance, we can write batched matrix multiply <code>bmm</code> by batching the <code>mm</code> operator.</p>
<p>It doesn't matter whether the implementation of the function uses dimension objects, it is also possible to add additional batch dimensions and then call a function:</p>
<p><code>py
def bmm(A, B):
    i = dims(1) # note: i here is a different value from i inside mm so it works
    return mm(A[i], B[i]).order(i)</code></p>
<p>The equivalent code in JAX, using <a href="https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#auto-vectorization-with-vmap">xmap or vmap</a> are transforms over functions. So there is a lot of syntactic distance between the specification of the dimension mappings, and the values where those mappings apply. Dims express the mapping as indexing of the tensor, right at the place where the function is being applied.</p>
<p><a href="https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html">xmap examples</a>:</p>
<p>```py
in_axes = [['inputs', 'hidden', ...],
           ['hidden', 'classes', ...],
           ['batch', 'inputs', ...],
           ['batch', ...]]</p>
<p>loss = xmap(named_loss, in_axes=in_axes, out_axes=[...])
print(loss(w1, w2, images, labels))
```</p>
<p>Equivalent with dimension objects:</p>
<p><code>py
batch, inputs, hidden, classes = dims(4)
print(loss(w1[inputs, hidden], w2[hidden, classes], images[batch, inputs], labels[batch],
      batch, inputs, hidden, classes))</code></p>
<h2>Composing matrix products, reshaping, and batching:</h2>
<p>Multi-headed attention is a good example of how these different uses compose. It reshapes the inputs, splitting out different attention heads. It batches over those attention heads, and it uses matrix products to compute attention scores.</p>
<p>```py
from torchdim import softmax
def multiheadattention(q, k, v, num_attention_heads, dropout_prob, use_positional_embedding):
    batch, query_sequence, key_sequence, heads, features = dims(5)
    heads.size = num_attention_heads</p>
<pre><code># binding dimensions, and unflattening the heads from the feature dimension
q = q[batch, query_sequence, [heads, features]]
k = k[batch, key_sequence, [heads, features]]
v = v[batch, key_sequence, [heads, features]]

# einsum-style operators to calculate scores,
attention_scores = (q*k).sum(features) * (features.size ** -0.5)

# use first-class dim to specify dimension for softmax
attention_probs = softmax(attention_scores, dim=key_sequence)

# dropout work pointwise, following Rule #1
attention_probs = torch.nn.functional.dropout(attention_probs, p=dropout_prob)

# another matrix product
context_layer = (attention_probs*v).sum(key_sequence)

# flatten heads back into features
return context_layer.order(batch, query_sequence, [heads, features])
</code></pre>
<p>```</p>
<h2>Indexing</h2>
<p>Rule #3 enables indexing because dimensions act as loop indices when used as a tensor. This allows for a lot of powerful behavior. The simplest might be using the dimensions to compute masks, such as extracting the upper triangular part of a matrix:</p>
<p><code>py
from torch import where
def triu(A):
    i,j = dims()
    a = A[i, j]
    return where(i &lt;= j, a, 0).order(i, j)
triu(torch.rand(3, 4))</code></p>
<p>Embedding bag does an embedding table lookup followed by a sum, which can be expressed concisely:</p>
<p>```py
def embedding_bag(input, embedding_weights):
    batch, sequence, features = dims(3)
    r = embedding_weights[input[batch, sequence], features].sum(sequence)
    return r.order(batch, features)</p>
<p>input = torch.tensor([[1, 0, 4, 3]])
W = torch.rand(5,2)
embedding_bag(input, W)
```</p>
<p>Relative positional embeddings associate an embedding vector with the distance between the query and the key in the sequence.
For instance, a key 3 and query 5 will have embedding ID <code>(5-3)=2</code>. We can use first-class dimensions to do the indexing arithmetic, and the embedding lookup:</p>
<p>```py
def relative_positional_embedding(q, k, distance_embedding_weight):
    batch, query_sequence, key_sequence, heads, features = dims(5)
    q = q[batch, query_sequence, [heads, features]]
    k = k[batch, key_sequence, [heads, features]]</p>
<pre><code>distance = query_sequence - key_sequence
n_embeddings = distance_embedding_weight.size(0)
index_bias = n_embeddings // 2

assert key_sequence.size + bias &lt;= n_embeddings

# indexing with dims
positional_embedding = distance_embedding_weight[distance + index_bias, features]

# matrix multiplies with dims
relative_position_scores_query = (q*positional_embedding).sum(features)
relative_position_scores_key = (k*positional_embedding).sum(features)
return  (relative_position_scores_query + relative_position_scores_key).order(batch, heads, key_sequence, query_sequence)
</code></pre>
<p>```</p>
<h1>Tensor Puzzlers</h1>
<p><a href="https://github.com/srush/Tensor-Puzzles">Tensor Puzzlers</a>, created by Sasha Rush, are a good exercise for learning the numpy and torch APIs by figuring out how to define common operations using a small set of primitive tensor operations.</p>
<p>However, the difficulty of many of the puzzlers lies not in how to compute the answer but the awkwardness of the primitives themselves.</p>
<p><strong>With first class dimensions, these puzzlers are nearly the same as the spec that defines them</strong></p>
<h3>Puzzle 3 - outer</h3>
<p>Compute <a href="https://numpy.org/doc/stable/reference/generated/numpy.outer.html">outer</a> - the outer product of two vectors.</p>
<p>```py
def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]</p>
<p>def outer(a, b):
    i, j = dims(2)
    return (a[i] * b[j]).order(i, j)
```</p>
<h3>Puzzle 4 - diag</h3>
<p>Compute <a href="https://numpy.org/doc/stable/reference/generated/numpy.diag.html">diag</a> - the diagonal vector of a square matrix.</p>
<p>```py
def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]</p>
<p>def diag(a):
    i = dims(1)
    return a[i, i].order(i)
```</p>
<h3>Puzzle 5 - eye</h3>
<p>Compute <a href="https://numpy.org/doc/stable/reference/generated/numpy.eye.html">eye</a> - the identity matrix.</p>
<p>```py
from torch import where
def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1</p>
<p>def eye(j: int):
    i,j = dims(sizes=[j, j])
    return where(i == j, 1, 0).order(i, j)
```</p>
<h3>Puzzle 6 - triu</h3>
<p>Compute <a href="https://numpy.org/doc/stable/reference/generated/numpy.triu.html">triu</a> - the upper triangular matrix.</p>
<p>```py
def triu_spec(out):
    for i in range(len(out)):
        for j in range(len(out)):
            if i &lt;= j:
                out[i][j] = 1
            else:
                out[i][j] = 0</p>
<p>def triu(j: int):
    i,j = dims(sizes=[j, j])
    return where(i &lt;= j, 1, 0).order(i, j)
```</p>
<h3>Puzzle 8 - diff</h3>
<p>Compute <a href="https://numpy.org/doc/stable/reference/generated/numpy.diff.html">diff</a> - the running difference.</p>
<p><code>py
def diff_spec(a, out):
    out[0] = a[0]
    for i in range(1, len(out)):
        out[i] = a[i] - a[i - 1]
def diff(a, i: int):
    i = dims(1)
    d = a[i] - a[i - 1]
    return where(i - 1 &gt;= 0, d, a[i]).order(i)</code></p>
<h3>Puzzle 9 - vstack</h3>
<p>Compute <a href="https://numpy.org/doc/stable/reference/generated/numpy.vstack.html">vstack</a> - the matrix of two vectors</p>
<p>```py
def vstack_spec(a, b, out):
    for i in range(len(out[0])):
        out[0][i] = a[i]
        out[1][i] = b[i]</p>
<p>def vstack(a, b):
    v, i = dims(sizes=[2, None])
    return where(v == 0,  a[i], b[i]).order(v, i)
```</p>
<h3>Puzzle 10 - roll</h3>
<p>Compute <a href="https://numpy.org/doc/stable/reference/generated/numpy.roll.html">roll</a> - the vector shifted 1 circular position.</p>
<p>```py
def roll_spec(a, out):
    for i in range(len(out)):
        if i + 1 &lt; len(out):
            out[i] = a[i + 1]
        else:
            out[i] = a[i + 1 - len(out)]</p>
<p>def roll(a, i: int):
    i = dims(sizes=[a.size(0)])
    return a[where(i + 1 &lt; i.size, i + 1, 0)].order(i)
```</p>
<h3>Puzzle 11 - flip</h3>
<p>Compute <a href="https://numpy.org/doc/stable/reference/generated/numpy.flip.html">flip</a> - the reversed vector</p>
<p>```py
def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]</p>
<p>def flip(a, i: int):
    i = dims(sizes=[a.size(0)])
    return a[i.size - i - 1].order(i)
```</p>
<h3>Puzzle 14 - sequence_mask</h3>
<p>Compute <a href="https://www.tensorflow.org/api_docs/python/tf/sequence_mask">sequence_mask</a> - pad out to length per batch.</p>
<p>```py
def sequence_mask_spec(values, length, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            if j &lt; length[i]:
                out[i][j] = values[i][j]
            else:
                out[i][j] = 0</p>
<p>def sequence_mask(values, length):
    j, i = dims()
    v = values[i, j]
    return where(j &lt; length[i], v, 0).order(i, j)
```</p>
<h1>Advantages of First-class Dimensions over String Dimensions</h1>
<p>The most prominent difference between named tensors using first-class dimensions and alternatives (einops, named tensors implemented in PyTorch today , <a href="https://nlp.seas.harvard.edu/NamedTensor">tensors considered harmful</a>, or xmap) is that dimensions are objects rather than strings. Using objects has a number of nice properties.</p>
<h3>Avoiding naming conflicts</h3>
<p>Using strings for dimensions introduces the possibility that two unrelated dimensions are given the same name. Using objects instead makes it clear the same names are not the same dimension. It's like the difference between having only global variables, and having the ability to locally bind names in functions.
 For instance, we defined <code>bmm</code> by batching a call to <code>mm</code>, and even though they both use the name <code>i</code> to identify a dimension.  Because each <code>i</code> is a different object, there is no naming conflict:</p>
<p>```py
def mm(A, B):
    i, j, k = dims()
    r = (A[i, k] * B[k, j]).sum(k)
    return r.order(i, j)</p>
<p>def bmm(A, B):
    i = dims() # note: doesn't matter than mm internally also uses i
    return mm(A[i], B[i])
```</p>
<p>Einops avoids conflicts by ensuring names are all introduced and removed in a single expression, but this precludes using long-lived dimensions to present implicit batching similar to xmap. When nested, JAX's xmap seems to consider axes the same if the string name matches. In the above example it would consider the <code>i</code> dimension to be the same dimension in both <code>bmm</code> and <code>mm</code> so the code would error.</p>
<h3>Reuse the same operator set</h3>
<p>Having a new object type allows us to extend the existing operator set of PyTorch rather than come up with new operators. For instance, binding dimensions using indexing follows semantically from Rules #1 and #3, so there is no need for a special operator to do binding. Even unbinding is just the <code>permute</code> operator which follows from Rule #2, though we call it <code>order</code> for clarity. In contrast, using strings requires coming up with new APIs such as <code>einsum</code> for matrix multiplies, or <code>rearrange</code> for doing permutations.</p>
<h3>Allows dims to act as tensors</h3>
<p>Rule #3 is not possible with strings since we cannot make strings behave as tensors. Without this rule, all of the indirect indexing that dims enable would not be easy to express.</p>
<h3>Dims can have methods</h3>
<p>For instance, as objects, dims can have a size, which allows us to do size inference of dimensions in various places in the API where string based APIs would have to take additional arguments specifying size.</p>
<h1>Comparison to tensor compilers or languages (e.g. TVM or Dex)</h1>
<p>The semantics and surface syntax of dimension objects resembles the kind of code written in tensor compilers such as <a href="https://halide-lang.org">Halide</a>, <a href="https://tvm.apache.org">TVM</a>, <a href="https://github.com/facebookresearch/TensorComprehensions">Tensor Comprehensions</a>, or the language <a href="https://github.com/google-research/dex-lang">Dex</a>.</p>
<p>These compilers and language have syntax and semantics that resemble the loop-level analogy similar to first-class dimensions. However, as compilers or statically typed languages, they require some binding code to go from running deep learning framework code in Python to using the compiled language. This often at least requires refactoring the compiled parts into their own functions, and may require defining a gradient function. Similar to graph mode frameworks, this adds friction to using and debugging the code.</p>
<p>Dimension objects are just an extension of the existing PyTorch tensors and eager semantics, so there is no friction switching between normal Python code and code that uses them. However, since loops over the dimensions are defined implicitly, they can still execute in Python with good performance compared to explicit loops. Furthermore, with dimension objects, a tensors containing dimensions can compute through code that is oblivious to the dimension such as batching examples. There is no need to separate code into 'compiled' vs 'eager'.</p>
<p>In this way, first-class dims are a way of adapting the nicer syntax of these array compilers and languages to eager numpy-style libraries.</p>
<h1>Performance Expectations</h1>
<p>First-class dimensions are not a compiler. They provide syntax for existing PyTorch operations such as advanced indexing that is easier to read and write. For large sized tensors, the performance of any statements including them will be the same as using the already existing operations. An important exception is the pattern matching of products and summation, where performance will be improved by issuing to a matrix-multiply kernel. The C++ implementation of dimensions adds a small overhead of around 2us on top of PyTorch's normal overhead of 8us to each function that uses them. In the future, the implementation can encorporate more fusion optimization to further improve performance of this style of code.</p>
<h2>License</h2>
<p>Functorch has a BSD-style license, as found in the <a href="LICENSE">LICENSE</a> file.</p>