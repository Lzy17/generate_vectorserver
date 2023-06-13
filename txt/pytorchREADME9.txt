<p>This directory contains the low-level tensor libraries for PyTorch,
as well as the new ATen C++ bindings.</p>
<p>The low-level libraries trace their lineage from the original Torch.  There are
multiple variants of the library, summarized here:</p>
<ul>
<li>TH = TorcH</li>
<li>THC = TorcH Cuda</li>
<li>THCS = TorcH Cuda Sparse (now defunct)</li>
<li>THNN = TorcH Neural Network (now defunct)</li>
<li>THS = TorcH Sparse (now defunct)</li>
</ul>
<p>(You'll also see these abbreviations show up in symbol names.)</p>
<h2>Reference counting</h2>
<p>PyTorch employs reference counting in order to permit tensors to provide
differing views on a common underlying storage.  For example, when you call
view() on a Tensor, a new THTensor is allocated with differing dimensions,
but it shares the same c10::StorageImpl with the original tensor.</p>
<p>Unfortunately, this means we are in the business of manually tracking reference
counts inside our C library code.  Fortunately, for most of our library code implementing
tensor operations, there is only one rule you have to remember:</p>
<blockquote>
<p><strong>Golden Rule of Reference Counting:</strong> You must either FREE or RETURN
a pointer which was returned by a function whose name begins with
<code>new</code> or which you called <code>retain</code> on.
If you return this pointer, your function name must begin with <code>new</code>.</p>
</blockquote>
<p>In a long function, there may be many invocations of functions with <code>new</code> in
their name.  Your responsibility is to go through each of them and ensure
that there is a matching <code>free</code> for it for EACH exit point of the function.</p>
<h3>Examples</h3>
<p>Suppose you want to get a reference to the indices of a sparse tensor.  This
function is called <code>newIndices</code>.  The <code>new</code> means you MUST free it when you're
done (usually at the end of your function.)  (It's worth noting that
<code>newIndices</code> doesn't actually allocate a fresh indices tensor; it just gives
you a pointer to the existing one.)  DO NOT directly access the member
variables of the struct.</p>
<p><code>THIndexTensor *indices = THSTensor_(newIndices)(state, sparse);
// ... do some stuff ...
THIndexTensor_(free)(state, indices);</code></p>
<p>Let's take a look at the implementation of <code>newIndices</code>.  This doesn't free the
return result of <code>newNarrow</code>, but returns it.  This justifies the <code>new</code> in its
name.</p>
<p><code>THIndexTensor *THSTensor_(newIndices)(const THSTensor *self) {
  // ...
  return THIndexTensor_(newNarrow)(self-&gt;indices, 1, 0, self-&gt;nnz);
}</code></p>
<p>Passing an object to another function does NOT absolve you of responsibility
of freeing it.  If that function holds on to a pointer to the object, it
will <code>retain</code> it itself.</p>
<p><code>THByteStorage *inferred_size = THByteStorage_newInferSize(size, numel);
  THTensor_(setStorage)(self, tensor-&gt;storage, tensor-&gt;storageOffset, inferred_size, NULL);
  c10::raw::intrusive_ptr::decref(inferred_size);</code></p>
<p>Sometimes, you have a tensor in hand which you'd like to use directly, but
under some conditions you have to call, e.g., <code>newContiguous</code>, to get it into
the correct form:</p>
<p><code>if (!(k_-&gt;stride(3) == 1) || !(k_-&gt;stride[2] == k_-&gt;size(3))) {
    kernel = THTensor_(newContiguous)(k_);
  } else {
    THTensor_(retain)(k_);
    kernel = k_;
  }
  ...
  c10::raw::intrusive_ptr::decref(kernel);</code></p>
<p>In this case, we have (redundantly) called <code>retain</code> on <code>k_</code>, so that we can
unconditionally free <code>kernel</code> at the end of the function; intuitively, you
want it to be possible to replace the conditional expression with an equivalent
function call, e.g., <code>kernel = THTensor_(newContiguous2D)(k_)</code>.</p>
<h3>Tips</h3>
<ul>
<li>
<p>If you have an early exit in a function (via a <code>return</code>), don't forget to
  <code>free</code> any pointers which you allocated up to this point.  If at all possible,
  move early exits prior to these allocations, so that you don't have to clean up.</p>
</li>
<li>
<p>Very occasionally, you may be able to implement an algorithm more efficiently
  if you "destroy" its input.  This is a <code>move</code>; after moving an object away,
  you must NOT <code>free</code> it.  This is the one exception to the rule, and at the
  moment there is only one instance of <code>move</code> in the code base.</p>
</li>
<li>
<p>We use <code>THError</code> to signal error cases, and fortunately,
  you do NOT need to make sure you've freed everything before calling <code>THError</code>,
  because by default, it aborts the entire process.  However, it's good style
  to call <code>THError</code> before performing any allocations, since in some cases we
  sketchily throw a C++ exception and try to recover (in particular, the test
  suite does this.)</p>
</li>
</ul>