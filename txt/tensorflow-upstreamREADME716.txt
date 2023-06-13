<h1>MLIR-HLO deallocation and buffer reuse passes</h1>
<p>MLIR-HLO deallocation is an alternative to the upstream buffer-deallocation and
buffer-hoisting passes.</p>
<p>The core concept is that of <em>ownership</em>, i.e. for each allocation, we track an
<em>ownership indicator</em> that can be moved around. These indicators can be
understood as a <code>std::unique_ptr</code> or alternatively a ref-counted pointer with a
maximum count of 1. At the end of a block, an ownership indicator must either
be yielded or the underlying alloc must be freed. In practice, it is not always
known whether a particular alloc is owned by the current block. Therefore, we
must also be able to represent empty ownership indicators (i.e., null pointers).</p>
<h2>Usage</h2>
<p>This is the recommended and supported pass pipeline to use these passes:</p>
<ol>
<li><code>hlo-split-alloc-tensors</code></li>
<li><code>one-shot-bufferize</code> with <code>create-deallocs=0</code></li>
<li><code>hlo-deallocate</code></li>
<li><code>hlo-deallocation-simplification</code></li>
<li><code>hlo-buffer-reuse</code></li>
<li><code>hlo-deallocation-simplification</code></li>
<li><code>hlo-deallocation-to-scf</code></li>
<li>(...)</li>
<li><code>convert-deallocation-ops-to-llvm</code></li>
</ol>
<p>It is possible to use just the deallocation pass or just buffer-reuse, but the
former isn't recommended because the output will be inefficient. The latter will
work as long as the invariants assumed by this code are maintained (in
particular, there should be no unranked memrefs in the input IR, since as
described above, the code here assigns special meaning to those).</p>
<h2>"ABI"</h2>
<p>As long as the IR contains only a single function, there shouldn't be any sharp
edges here. If there are multiple functions, it is important to pay attention to
the ABI assumed here:</p>
<ol>
<li>Function arguments are always owned by the caller.</li>
<li>Function results are always owned by the caller <strong>and do not alias with any
    function arguments</strong>. In other words, function results are always freshly
    allocated buffers. Function arguments may alias each other.</li>
</ol>
<p>Warning: The second condition here is particularly important - if a function
returns one of its arguments, the deallocation pass will silently introduce a
double free.</p>
<p>This restriction could be lifted by introducing ownership indicators for
function arguments, but as of March 2023, this is not done.</p>
<h2>The deallocation pass</h2>
<p>The deallocation pass assumes that:</p>
<ol>
<li>The input IR was fully bufferized (i.e., no tensors are left in the
    program).</li>
<li>No <code>dealloc</code>s, <code>alloca</code>s or <code>realloc</code>s exist yet.</li>
<li>No <code>memrefs</code> with distinct element types alias (strict aliasing; in
    particular, no <code>xla_cpu.memref_element_cast</code> ops should exist at this point)</li>
</ol>
<p>The basic deallocation algorithm works mostly locally within blocks. It
transforms the input IR op by op, keeping track of memref alias information as
it goes. For each op, it produces the following information: 1) which allocs
were released by the parent block (i.e., are no longer owned by it; more on that
in the section on transferring ownership), 2) which new allocs are now owned by
the parent block. For example, when processing an <code>alloc</code> op, nothing is
released, and the result of the op is now owned by the block. It also keeps
track of aliasing information. Conservatively, it is assumed that all inputs
alias all compatible outputs.</p>
<p>When transforming a block, it is not possible to know in general whether
<code>memref</code> arguments are owned by it or by some ancestor. Therefore, we introduce
ownership indicator arguments (<code>!deallocation.ownership</code>) for each <code>memref</code>
argument. Inside the block, <code>allocs</code> and alias sets are tracked as described
above. At the end of the block, we must reconcile these memrefs and potentially
owned allocs. We can do this separately for those that are yielded from the
block and those that aren't.</p>
<p>For <code>memrefs</code> (or rather sets of <code>memrefs</code> that potentially alias) that aren't
yielded, we must free the corresponding <code>alloc</code> if we own it. In general, we
can't know statically whether that's the case, so we use the <code>retain</code> op, which
frees non-null allocs [^1] that are no longer needed. To find the place to
insert the op, we simply traverse the block backwards, starting from the
terminator, and look for the last op that contains any reference to a memref
from the alias set.</p>
<p><code>// Free %alloc_0 and %alloc_1 iff they are non-null.
  deallocation.retain() of(%alloc_0, %alloc_1)
      : (!deallocation.ownership, !deallocation.ownership) -&gt; ()</code></p>
<p>For <code>memrefs</code> that are yielded, we also insert retain ops, but this time, we
must retain allocs if we own them. The <code>retain</code> ops look like this:</p>
<p><code>// Check if %yielded_memref aliases with any of %a, %b or %c. If it does,
  // return the corresponding memref. Free the others if they are non-null.
  %maybe_owned = deallocation.retain(%yielded_memref) of(%a, %b, %c)
      : (!deallocation.ownership, !deallocation.ownership, !deallocation.ownership)
      -&gt; (!deallocation.ownership)</code></p>
<p>To understand where such ops come from, consider the following code:</p>
<p><code>%result = scf.if %cond -&gt; memref&lt;2xi32&gt; {
    scf.yield %some_alloc : memref&lt;2xi32&gt;
  } else {
    %new_alloc = memref.alloc() : memref&lt;2xi32&gt;
    scf.yield %new_alloc : memref&lt;2xi32&gt;
  }</code></p>
<p>Whether the parent block owns the alloc that backs <code>%result</code> depends on which
branch was taken. Therefore, after transforming the block, the <code>if</code> will look
like this:</p>
<p><code>%result, %result_ownership = scf.if %cond -&gt; memref&lt;2xi32&gt; {
    %null = deallocation.null
    scf.yield %some_alloc, %null : memref&lt;2xi32&gt;, !deallocation.ownership
  } else {
    %new_alloc = memref.alloc() : memref&lt;2xi32&gt;
    %new_alloc_owned = deallocation.own %new_alloc : memref&lt;2x32&gt;
    scf.yield %new_alloc, %new_alloc_owned : memref&lt;2xi32&gt;, !deallocation.ownership
  }</code></p>
<p><code>%result_ownership</code> is nonnull iff <code>%result</code> is owned by the parent block. If
<code>%result</code> is yielded, the corresponding retain op would be:</p>
<p><code>%yielded_result_ownership = deallocation.retain(%result) of(%result_ownership)</code></p>
<p>However, here we can statically determine that this always results in
<code>%result_ownership</code>, so the <code>retain</code> op will not be emitted.</p>
<h3>Loops and if: <code>RegionBranchOpInterface</code></h3>
<p>RegionBranchOpInterface ops mostly follow what was described above for blocks,
but there are two interesting things about them:</p>
<ol>
<li>Regions with multiple predecessors</li>
<li>Transferring ownership to the op</li>
</ol>
<p><em>Multiple predecessors</em>. In <code>scf.while</code>, and <code>scf.if</code>, some regions have
multiple predecessors (in the case of <code>while</code>, the <code>before</code> region, in the case
of <code>if</code>, the parent region). As it turns out, no special logic is required to
handle this - the regions will always yield the same types of memrefs, and
therefore the added ownership indicators will also have the same types.</p>
<p><em>Transfer of ownership</em>. If a <code>memref</code> operand of a loop has no further uses
after the loop, we can transfer the ownership indicator for the operand to the
loop. Note that this does not necessarily mean ownership is actually
transferred - the ownership indicator may be null.</p>
<h4>Implicit capture / implicit transfer of ownership</h4>
<p>Consider the following program, which conditionally reallocates a memref:</p>
<p><code>%alloc = memref.alloc(%size) : memref&lt;?xi32&gt;
scf.for %i = %lb to %ub step %step iter_args(%arg0 = %alloc) {
  %should_grow, %new_size = "dummy.check_capacity"(%arg0)
    : (memref&lt;?xi32&gt;) -&gt; (i1, index)
  %mem = scf.if %should_grow {
    %0 = memref.realloc %arg0(%new_size) : memref&lt;?xi32&gt; -&gt; memref&lt;?xi32&gt;
    scf.yield %0 : memref&lt;?xi32&gt;
  } else {
    scf.yield %arg0 : memref&lt;?xi32&gt;
  }
  "dummy.use"(%mem) : (memref&lt;?xi32&gt;) -&gt; ()
  scf.yield %mem : memref&lt;?xi32&gt;
}</code></p>
<p><code>%arg0</code> is owned by the loop, but it must not be deallocated at the end of the
loop body - otherwise, we'd run into a double free when it is reallocated.</p>
<p>We solve this by defining implicit captures, or implicit transfer of ownership.
<code>memref.realloc</code> ops are considered to implicitly capture and release their
operand. There are a couple of restrictions to this:</p>
<ol>
<li>Only ops owned by the parent block can be implicitly captured.</li>
<li>Implicit capture is only allowed in <code>scf.if</code> ops. This rule may be applied
    recursively.</li>
<li>The implicit capture must be the last use of the captured value across all
    execution paths.</li>
<li>Implied by the previous rule: Implicit capture is not allowed in <code>scf.if</code>
    ops that do not have an else branch.</li>
</ol>
<p>To illustrate these restrictions, we can look at some IR that violates them:</p>
<p><code>%alloc = memref.alloc()
scf.if %cond {
  %0 = memref.realloc %alloc  // invalid
}</code></p>
<p>This IR contains an implicit capture inside an <code>scf.if</code> without an <code>else</code>
branch. Since <code>%alloc</code> is only freed if <code>%cond</code> is true, there must be some
further use of <code>%alloc</code>, which is invalid. To make this valid, the following IR
should be emitted instead:</p>
<p><code>%alloc = memref.alloc()
%0 = scf.if %cond {
  %1 = memref.realloc %alloc
  scf.yield %1
} else {
  scf.yield %alloc
}</code></p>
<p>Note that <code>scf.yield %alloc</code> is executed no execution path that also executes
the <code>realloc</code>, so condition 3 is not violated.</p>
<p>An example that violates condition 1:</p>
<p><code>%alloc = memref.alloc()
scf.for %i = %lb to %ub step %step {
  scf.if ... {
    %0 = memref.realloc %alloc  // invalid
  } else {
    ...
  }
}</code></p>
<p><code>%alloc</code> cannot be implicitly captured here, since there is no chain of ancestor
<code>scf.if</code> ops to its definition. To make this valid, turn <code>%alloc</code> into an
<code>iter_arg</code>:</p>
<p><code>%alloc = memref.alloc()
%0 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %alloc) {
  %1 = scf.if ... {
    %2 = memref.realloc %alloc
  } else {
    ...
  }
  scf.yield %1
}</code></p>
<h2>Ops in the deallocation dialect</h2>
<h3>The <code>null</code> op</h3>
<p>Creates a null pointer.</p>
<h3>The <code>own</code> op</h3>
<p>Declares ownership of an alloc and returns an ownership indicator. This is
lowered to an extraction of the alloc's base pointer.</p>
<h3>The <code>retain</code> op</h3>
<p>Takes a list of memrefs and a list of ownership indicator. For each memref,
returns the ownership (alloc) that it was derived from (if present). Each alloc
is returned at most once. Alloc that are not returned are freed.</p>
<p>Some retain ops can be simplified to a no op (e.g. if there's only one alloc
and one memref, and they're the same). Others can be rewritten to memref.dealloc
(if we know that the alloc is non-null and there is no memref). This is done by
the <code>deallocation-simplification</code> pass.</p>
<p>There are two lowerings of <code>retain</code>: retains with a single memref or a single
ownership indicator are lowered to a sequence of <code>scf.if</code> ops. Lowerings with
more than one of either are instead lowered to a library call. For details, see
the section on the deallocation-to-scf pass.</p>
<h3>The <code>get_buffer</code> op</h3>
<p>Returns the memref's base pointer as an index.</p>
<h2>The buffer reuse pass</h2>
<p>The buffer reuse pass is intended to be run after the deallocation pass and
assumes that the code has the structure that the pass guarantees (in particular,
unranked memref == ownership indicator). For best results, the IR should be
canonicalized first.</p>
<h3>Loop simplification</h3>
<p>As a preprocessing step, this pass transforms <code>retain</code> ops that operate on the
result of loops. Consider the following IR:</p>
<p><code>%alloc1 = memref.alloc() : memref&lt;4xi32&gt;
%alloc2 = memref.alloc() : memref&lt;4xi32&gt;
%0:4 = scf.while(%arg0 = %alloc1, $arg1 = %alloc2) {
  scf.condition(%cond) %arg1, %arg0
do {
  (...)
  scf.yield %arg0, %arg1
}
memref.dealloc %0#0 : memref&lt;4xi32&gt;
memref.dealloc %0#1 : memref&lt;4xi32&gt;</code></p>
<p><code>%0#0</code> and <code>%0#1</code> are <code>%alloc1</code> and <code>%alloc2</code>, in some order. Since there is no
further use of these allocs and they are all deallocated, we can rewrite the
operands to <code>%alloc1</code> and <code>%alloc2</code>, even though we don't know which one is
which.</p>
<p>The purpose of this preprocessing step is to allow more buffer reuse, which
requires <code>dealloc</code>/<code>alloc</code> pairs to work.</p>
<h3>Buffer reuse</h3>
<p>Buffer reuse coalesces <code>dealloc</code>/<code>alloc</code> pairs:</p>
<p><code>memref.dealloc %alloc : memref&lt;100xi32&gt;
(...)
%alloc_1 = memref.alloc() : memref&lt;100xi32&gt;</code></p>
<p>Instead of deallocating and allocating, we replace all uses of <code>%alloc_1</code> with
<code>%alloc</code>. Currently, we only do this for immediate <code>dealloc</code>/<code>alloc</code> pairs with
no other <code>alloc</code>/<code>dealloc</code> ops in between. So in the example above, if <code>(...)</code>
included any other allocation or deallocation, no reuse would occur.</p>
<h3>Copy elision</h3>
<p>Another simple transformation eliminates <code>alloc</code>/<code>copy</code>/<code>dealloc</code> patterns:</p>
<p><code>%a = memref.alloc() : memref&lt;100xi32&gt;
(... 1)  // no uses of %a
memref.copy %b, %a : memref&lt;100xi32&gt; to memref&lt;100xi32&gt;
memref.dealloc %b : memref&lt;100xi32&gt;
(... 2)  // uses of %a</code></p>
<p>Since <code>%a</code> is completely overwritten with <code>%b</code>, which is deallocated immediately
afterwards, we can remove the allocation of <code>%a</code> and replace its uses with <code>%b</code>.</p>
<p><code>(... 1)  // potential uses of %b
(... 2)  // all uses of %a replaced with %b</code></p>
<p>Note: This pattern could be generalized to only look at copy ops and the uses of
its operand, leaving the elimination of the allocation and deallocation to other
patterns. As of March 2023, this is not done.</p>
<h3>Hoisting</h3>
<p>The second transformation implemented in this pass is buffer hoisting. This
simply looks for allocs that happen in each iteration of a loop and moves them
out of the loop:</p>
<p><code>scf.for %i = %c0 to %c1000 step %c1 {
  %foo = memref.alloc() : memref&lt;100xi32&gt;
  (...)
  memref.dealloc %foo : memref&lt;100xi32&gt;
}</code></p>
<p>Since the contents of a freshly allocated memref are undefined, this can be
transformed as follows:</p>
<p><code>%foo = memref.alloc() : memref&lt;100xi32&gt;
scf.for %i = %c0 to %c1000 step %c1 {
  (...)
}
memref.dealloc %foo : memref&lt;100xi32&gt;</code></p>
<p>The same transformation applies for while loops, with the caveat that it may
increase peak heap usage in that case.</p>
<h3>Double buffering</h3>
<p>Double buffering can be considered a variant of hoisting. It is useful in cases
where use ranges of buffers overlap, preventing simple hoisting. Consider the
following IR (ownership indicator omitted for clarity):</p>
<p><code>%0 = scf.for %i = %c0 to %c1000 step %c1 iter_args(%arg = %alloc)
    -&gt; memref&lt;100xi32&gt; {
  %tmp = memref.alloc() : memref&lt;100xi32&gt;
  "some.op"(%tmp, %arg) : (memref&lt;100xi32&gt;, memref&lt;100xi32&gt;) -&gt; ()
  memref.dealloc %arg : memref&lt;100xi32&gt;
  scf.yield %tmp : memref&lt;100xi32&gt;
}
memref.dealloc %0 : memref&lt;100xi32&gt;</code></p>
<p>The live ranges of <code>%alloc</code> and <code>%tmp</code> overlap, so we can't do straightforward
hoisting here. However, we only need two distinct buffers at any given time, so
instead, we introduce an additional iter arg for the temporary buffer, hoist and
swap in each iteration:</p>
<p><code>%tmp = memref.alloc() : memref&lt;100xi32&gt;
%0, %1 = scf.for %i = %c0 to %c1000 step %c1
    iter_args(%arg = %alloc, %tmp_ = %tmp) -&gt; memref&lt;100xi32&gt; {
  "some.op"(%tmp_, %arg) : (memref&lt;100xi32&gt;, memref&lt;100xi32&gt;) -&gt; ()
  scf.yield %tmp_, %arg : memref&lt;100xi32&gt;, memref&lt;100xi32&gt;
}
memref.dealloc %1 : memref&lt;100xi32&gt;
memref.dealloc %0 : memref&lt;100xi32&gt;</code></p>
<p>Note that the presence of a deallocation of <code>%arg</code> inside the loop implies no
further uses of <code>%alloc</code> after the loop. So, similarly to the case described in
the section on loop simplification, it doesn't matter which alloc is in <code>%0</code> and
which one is in <code>%1</code>.</p>
<p>Double buffering works analogously for <code>while</code> loops, with the exception that
buffers have to be plumbed through the before region.</p>
<p>Note: as of March 2023, double buffering allocations in <code>while</code> loops is only
implemented for the <code>after</code> region.</p>
<h2>The split-alloc-tensors pass</h2>
<p>This pass is a helper pass to improve the behavior of the other passes when used
together with <code>one-shot-bufferize</code>. The purpose of this pass is to prevent
accidental buffer reuse by <code>one-shot-bufferize</code> by ensuring each <code>alloc_tensor</code>
is used only once, thereby minimizing the sizes of live ranges and enabling the
buffer reuse pass to work optimally.</p>
<h2>The deallocation-to-scf pass</h2>
<p>As described previously, most <code>deallocation.retain</code> ops are eliminated either by
canonicalization or by <code>buffer-reuse</code>. <code>deallocation-to-scf</code> lowers the ones
that remain to sequences of <code>scf.if</code> ops.</p>
<p>Because the size of the emitted code is in <code>O(|allocs| * |memrefs|)</code>, we only
use this lowering when at least one of <code>|allocs|</code> or <code>|memrefs|</code> is 1.</p>
<p>[^1]: <code>memref.dealloc</code> happens to tolerate null inputs as well, but at this
    point of the pipeline, we assume that the argument is always non-null,
    because 1) this behavior isn't documented 2) it simplifies analysis in
    subsequent passes.</p>