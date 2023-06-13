<h1>SparseTensor</h1>
<p>Sparse Tensors are stored as two dense tensors and a shape:</p>
<ul>
<li><code>indices</code>: a <code>brain::Tensor</code> storing a matrix, typically <code>int64</code></li>
<li><code>values</code>: a <code>brain::Tensor</code> storing a vector with values of type T.</li>
<li><code>shape</code>: a <code>TensorShape</code> storing the bounds of the underlying tensor</li>
<li><code>order</code>: (optional) a <code>gtl::InlinedVector&lt;int64,8&gt;</code> with the dimensions
            along which the indices are ordered.</li>
</ul>
<p>Let</p>
<pre><code>ix = indices.matrix&lt;int64&gt;()
vals = values.vec&lt;T&gt;()
</code></pre>
<p>The shape of <code>ix</code> is <code>N x NDIMS</code>, and each row corresponds to the
index of a single element of the sparse tensor.</p>
<p>The length of <code>vals</code> must be <code>N</code>, and <code>vals(i)</code> corresponds to the
value with index <code>ix(i,:)</code>.</p>
<p>Shape must be a <code>TensorShape</code> with <code>dims() == NDIMS</code>.
The shape is the full shape of the dense tensor these indices
represent.</p>
<p>To be specific, the representation (pseudocode) is:</p>
<pre><code>tensor[ix[i,:]] == vals[i] for i = 0, ..., N-1
</code></pre>
<h2>Ordering</h2>
<p>Indices need not be provided in order.  For example, the following
index matrix is ordered according to dimension order <code>{0, 1, 2}</code>.</p>
<pre><code>[0 0 1]
[0 1 1]
[2 0 2]
</code></pre>
<p>However, you can provide an unordered version:</p>
<pre><code>[2 0 2]
[0 0 1]
[0 1 1]
</code></pre>
<p>If the SparseTensor is constructed without a provided order, then a
the default order is <code>{-1, ..., -1}</code>.  Certain operations will fail or crash
when the order is not provided.</p>
<p>Resorting the SparseTensor in-place (which resorts the underlying index and
values tensors in-place) will update the order.  The cost of reordering the
matrix is <code>O(N*log(N))</code>, and requires <code>O(N)</code> additional temporary space to store
a reordering index.  If the default order is not specified and reordering is not
performed, the following will happen:</p>
<ul>
<li><code>group()</code> will <strong>raise an assertion failure</strong></li>
<li><code>IndicesValid()</code> will <strong>raise an assertion failure</strong></li>
</ul>
<p>To update the internal index ordering after construction, call
<code>Reorder&lt;T&gt;()</code> via, e.g., <code>Reorder&lt;T&gt;({0,1,2})</code>.
After this step, all the above methods should work correctly.</p>
<p>The method <code>IndicesValid()</code> checks to make sure:</p>
<ul>
<li><code>0 &lt;= ix(i, d) &lt; shape.dim_size(d)</code></li>
<li>indices do not repeat</li>
<li>indices are in order</li>
</ul>
<h2>Iterating</h2>
<h3>group({grouping dims})</h3>
<ul>
<li>provides an iterator that groups entries according to
   dimensions you care about</li>
<li>may require a sort if your data isn't presorted in a way that's
   compatible with grouping_dims</li>
<li>for each group, returns the group index (values of the group
   dims for this iteration), the subset of indices in this group,
   and the subset of values in this group.  these are lazy outputs
   so to read them individually, copy them as per the example
   below.</li>
</ul>
<h4><strong>NOTE</strong></h4>
<p><code>group({dim0, ..., dimk})</code> will <strong>raise an assertion failure</strong> if the
order of the SparseTensor does not match the dimensions you wish to group by.
You must either have your indices in the correct order and construct the
SparseTensor with</p>
<pre><code>order = {dim0, ..., dimk, ...}
</code></pre>
<p>or call</p>
<pre><code>Reorder&lt;T&gt;({dim0, .., dimk, ...})
</code></pre>
<p>to sort the SparseTensor before grouping.</p>
<p>Example of grouping:</p>
<pre><code>Tensor indices(DT_INT64, TensorShape({N, NDIMS});
Tensor values(DT_STRING, TensorShape({N});
TensorShape shape({dim0,...});
SparseTensor sp(indices, vals, shape);
sp.Reorder&lt;tstring&gt;({1, 2, 0, 3, ...}); // Must provide NDIMS dims.
// group according to dims 1 and 2
for (const auto&amp; g : sp.group({1, 2})) {
  cout &lt;&lt; "vals of ix[:, 1,2] for this group: "
       &lt;&lt; g.group()[0] &lt;&lt; ", " &lt;&lt; g.group()[1];
  cout &lt;&lt; "full indices of group:\n" &lt;&lt; g.indices();
  cout &lt;&lt; "values of group:\n" &lt;&lt; g.values();

  TTypes&lt;int64&gt;::UnalignedMatrix g_ix = g.indices();
  TTypes&lt;tstring&gt;::UnalignedVec g_v = g.values();
  ASSERT(g_ix.dimension(0) == g_v.size());  // number of elements match.
}
</code></pre>
<h2>ToDense</h2>
<p>Converts sparse tensor to dense.  You must provide a pointer to the
dense tensor (preallocated).  <code>ToDense()</code> will optionally
preinitialize the tensor with zeros.</p>
<p>Shape checking is performed, as is boundary checking.</p>
<pre><code>Tensor indices(DT_INT64, TensorShape({N, NDIMS});
Tensor values(DT_STRING, TensorShape({N});
TensorShape shape({dim0,...});
SparseTensor sp(indices, vals, shape);
ASSERT(sp.IndicesValid());  // checks ordering &amp; index bounds.

Tensor dense(DT_STRING, shape);
// initialize other indices to zero.  copy.
ASSERT(sp.ToDense&lt;tstring&gt;(&amp;dense, true));
</code></pre>
<h2>Concat</h2>
<p>Concatenates multiple SparseTensors and returns a new SparseTensor.
This concatenation is with respect to the "dense" versions of these
SparseTensors.  Concatenation is performed along dimension order[0]
of all tensors.  As a result, shape[order[0]] may differ across
the inputs, but shape[d] for d != order[0] must match across all inputs.</p>
<p>We call order[0] the <strong>primary dimension</strong>.</p>
<p><strong>Prerequisites</strong></p>
<ul>
<li>The inputs' ranks must all match.</li>
<li>The inputs' order[0] must all match.</li>
<li>The inputs' shapes must all match except for dimension order[0].</li>
<li>The inputs' values must all be of the same type.</li>
</ul>
<p>If any of these are false, concat will die with an assertion failure.</p>
<p>Example:
Concatenate two sparse matrices along columns.</p>
<p>Matrix 1:</p>
<pre><code>[0 0 1]
[2 0 0]
[3 0 4]
</code></pre>
<p>Matrix 2:</p>
<pre><code>[0 0 0 0 0]
[0 1 0 0 0]
[2 0 0 1 0]
</code></pre>
<p>Concatenated Matrix:</p>
<pre><code>[0 0 1 0 0 0 0 0]
[2 0 0 0 1 0 0 0]
[3 0 4 2 0 0 1 0]
</code></pre>
<p>Expected input shapes, orders, and <code>nnz()</code>:</p>
<pre><code>shape_1 = TensorShape({3, 3})
shape_2 = TensorShape({3, 8})
order_1 = {1, 0}  // primary order is 1, columns
order_2 = {1, 0}  // primary order is 1, must match
nnz_1 = 4
nnz_2 = 3
</code></pre>
<p>Output shapes and orders:</p>
<pre><code>conc_shape = TensorShape({3, 11})  // primary dim increased, others same
conc_order = {1, 0}  // Orders match along all inputs
conc_nnz = 7  // Sum of nonzeros of inputs
</code></pre>
<p>Coding Example:</p>
<pre><code>Tensor ix1(DT_INT64, TensorShape({N1, 3});
Tensor vals1(DT_STRING, TensorShape({N1, 3});
Tensor ix2(DT_INT64, TensorShape({N2, 3});
Tensor vals2(DT_STRING, TensorShape({N2, 3});
Tensor ix3(DT_INT64, TensorShape({N3, 3});
Tensor vals3(DT_STRING, TensorShape({N3, 3});

SparseTensor st1(ix1, vals1, TensorShape({10, 20, 5}), {1, 0, 2});
SparseTensor st2(ix2, vals2, TensorShape({10, 10, 5}), {1, 0, 2});
// For kicks, st3 indices are out of order, but order[0] matches so we
// can still concatenate along this dimension.
SparseTensor st3(ix3, vals3, TensorShape({10, 30, 5}), {1, 2, 0});

SparseTensor conc = SparseTensor::Concat&lt;string&gt;({st1, st2, st3});
Tensor ix_conc = conc.indices();
Tensor vals_conc = conc.values();
EXPECT_EQ(conc.nnz(), st1.nnz() + st2.nnz() + st3.nnz());
EXPECT_EQ(conc.Shape(), TensorShape({10, 60, 5}));
EXPECT_EQ(conc.Order(), {-1, -1, -1});

// Reorder st3 so all input tensors have the exact same orders.
st3.Reorder&lt;tstring&gt;({1, 0, 2});
SparseTensor conc2 = SparseTensor::Concat&lt;string&gt;({st1, st2, st3});
EXPECT_EQ(conc2.Order(), {1, 0, 2});
// All indices' orders matched, so output is in order.
EXPECT_TRUE(conc2.IndicesValid());
</code></pre>