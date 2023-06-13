<h1>Google ML Structured Dialect</h1>
<p>The <code>gml_st</code> dialect will contain a loop-like construct and subset operations
that should allow support for fusion beyond rectangular tiles. This is necessary
for operations like <code>gather</code>, <code>scatter</code>, <code>concat</code> and more.</p>
<h2>Overview</h2>
<h3>Tiling and fusion</h3>
<p>Tiling of an op is performed by creating a loop that computes subsets of the
result. Usually the tiling is needed to enable vectorization or distribution.</p>
<p>Before tiling</p>
<p><code>%0 = op(%input)</code></p>
<p>After tiling</p>
<p><code>loop (%ivs)
  %1 = subset(%input, %ivs)
  %2 = op (%1)</code></p>
<p>Fusion of a producer op into a tiled consumer consists of two main parts:
computing subsets of producer's operands and moving the producer op into the
loop body so that it operates on the subsets of its original operands.</p>
<p>After consumer tiling
<code>%0 = producer (%input)
loop (%ivs)
  %1 = subset(%0, %ivs)
  %2 = consumer(%1)</code></p>
<p>After producer fusion</p>
<p><code>loop (%ivs)
  %0 = subset(%input, %ivs)
  %1 = producer(%0)
  %2 = consumer (%1)</code></p>
<p>There is some duality between tiling and fusion. One can consider tiling as
fusion of the op into a loop that partitions the iteration space and just
returns identity for every subset. On the other hand, fusion can be seen as
tiling of the producer and then merging of the loop bodies.</p>
<h3>Subset operations</h3>
<p>Linalg has support for hyperrectangular subsets (tiles) of tensor/memref
operands. Currently, Linalg's fusion assumes that the tiling is performed only
using <code>tensor.extract_slice/tensor.insert_slice</code> and <code>memref.subview</code>
operations.
There are several disadvantages to that approach:</p>
<p>If some of the operands are not affected by tiling, i.e. the tiling was
performed along dimensions that are not present in the operand, then we cannot
fuse anymore the producer of the operand. That can happen when <code>linalg.generic</code>
broadcasts one of the operands or when the output is tiled, but not the
reduction dimensions</p>
<p>Support for fusion with ops like <code>gather</code>, <code>scatter</code>, <code>concat</code> for some of the
cases can only be done via <code>TilingInterface</code>
(<a href="https://llvm.discourse.group/t/rfc-for-tilinginterface-for-tiling-operations-that-dont-fit-into-linalg-structured-operation-definition/3897/7">RFC</a>).</p>
<p><strong>Example of a tiled op</strong></p>
<p><code>%sum = linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%c80, %c60) step (%c4, %c4)
          ins (%in_ = %in: tensor&lt;80x60xf32&gt;, %cst_ = %cst: f32)
          outs (%out_ = %out: tensor&lt;80xf32&gt;)
          iterators["parallel", "reduction"] {
  %in_sub = tensor.extract_slice %in_[%i, %j] [4, 4] [1, 1]
      : tensor&lt;80x60xf32&gt; to tensor&lt;4x4xf32&gt;
  %out_sub = tensor.extract_slice %out_[%i] [4] [1]
      : tensor&lt;80xf32&gt; to tensor&lt;4xf32&gt;
  %reduction = linalg.generic {
      indexing_maps = [affine_map&lt;(d0, d1) -&gt; (d0, d1)&gt;,
                       affine_map&lt;(d0, d1) -&gt; (d0)&gt;],
      iterator_types = ["parallel", "reduction"]}
      ins(%in_sub : tensor&lt;4x4xf32&gt;)
      outs(%out_sub : tensor&lt;4xf32&gt;) {
    ^bb0(%a: f32, %b: f32):
      %0 = arith.addf %a, %b : f32
      linalg.yield %0 : f32
  } -&gt; tensor&lt;4xf32&gt;
  %update = tensor.insert_slice %reduction into %out_[%i] [4] [1]
      : tensor&lt;4xf32&gt; into tensor&lt;80xf32&gt;
  linalg.yield %update : tensor&lt;80xf32&gt;
}</code></p>
<p>The body of this loop models read-modify-write of the output tensor. The tile
that we extract from <code>%out_</code> should have the same sizes/offsets/strides as the
destination of <code>tensor.insert_slice</code>. The arguments of <code>tensor.extract_slice</code>
and <code>tensor.insert_slice</code> are currently not required to encode the same tile.</p>
<p>We introduce new operations that define subsets on tensors/memrefs</p>
<ul>
<li><code>subset.full %tensor</code> - the subset spans the original tensor fully</li>
<li><code>subset.tile %tensor [%offsets][%sizes][%strides]</code> - defines a rectangular
   tile</li>
<li><code>subset.filter %tensor[%indices]</code> - the subset has the same shape as the
   original tensor, but only the values at %indices are populated. This can be a
   sparse tensor.</li>
<li><code>subset.point %tensor[%index]</code> - the subset contains a single element</li>
</ul>
<h3>Structured loop</h3>
<p>We introduce <code>gml_st.loop</code> that keeps the subset definition separately from the
materialization.</p>
<p><code>linalg.generic</code> has <code>AffineMap</code> attributes that specify the indexing maps and a
region that models the computation on the element types of the operand
tensors/memrefs. The region ends with <code>linalg.yield</code> terminator that yields the
element of the output. The load and store ops in that case are implicit, so
are extraction/insertion in <code>gml_st.loop</code>.</p>
<p><code>gml_st.loop</code> has one region that contains subset operations to define the
dense/sparse ranges that we are working with and also <code>gml_st.materialize</code> ops
to convert subset spec to a tensor or memref.</p>
<p><code>gml_st.yield</code> is the terminator for <code>gml_st.loop</code> that takes computed tensors
and a subset specification for which the computation was done. Note that this
way we don't have to explicitly write a destructive update with
<code>tensor.insert_slice</code> and then yield a full tensor. Here, we yield values for a
subset.</p>
<p>```
%sum = gml_st.loop (%i, %j) = (%c0, %c0) to (%c80, %c60) step (%c4, %c4)
           ins (%in_ = %in: tensor&lt;80x60xf32&gt;, %cst_ = %cst: f32)
           outs (%out_ = %out: tensor&lt;80xf32&gt;)
           iterators["parallel", "sequential"] {
  %in_tile = gml_st.tile %in_[%i, %j] [4, 4] [1, 1]
      : tensor&lt;80x60xf32&gt; to !gml_st.subset&lt;4x4xf32&gt;
  %out_tile = gml_st.tile %out_[%i] [4] [1]
      : tensor&lt;80xf32&gt; to !gml_st.subset&lt;4xf32&gt;</p>
<p>%in_sub = gml_st.materialize %in_tile
      : !gml_st.subset&lt;4x4xf32&gt; to tensor&lt;4x4xf32&gt;
  %out_sub = gml_st.materialize %in_tile
      : !gml_st.subset&lt;4xf32&gt; to tensor&lt;4xf32&gt;
  %reduction = linalg.generic {
      indexing_maps = [affine_map&lt;(d0, d1) -&gt; (d0, d1)&gt;,
                       affine_map&lt;(d0, d1) -&gt; (d0)&gt;],
      iterator_types = ["parallel", "reduction"]}
      ins(%in_sub : tensor&lt;4x4xf32&gt;)
      outs(%out_sub : tensor&lt;4xf32&gt;) {
    ^bb0(%a: f32, %b: f32):
      %0 = arith.addf %a, %b : f32
      linalg.yield %0 : f32
  } -&gt; tensor&lt;4xf32&gt;
  gml_st.yield %reduction to %out_tile
      : tensor&lt;4xf32&gt; to !gml_st.subset&lt;4xf32&gt;
}
```</p>
<p>Currently, tiling of the consumer and fusion of its producers are tightly
coupled. If the fusion is happening not in the same pass, then some analysis is
required to find the [consumer - <code>tensor.extract_slice</code> - producer] triple to
perform the fusion. Keeping the subset computations separately from the
"compute" ops not only improves readability but also simplifies fusion, since we
have a subset computation per operand and we can just specify what argument of
the loop we want to fuse.</p>
<p>It also simplifies the bufferization, since we don't need to introduce the
additional operations in MemRef dialect for every subset operation in TensorOps.</p>