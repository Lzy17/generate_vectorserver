<h1>Sparse benchmarks</h1>
<p>These sets of benchmarks are for the sparse matrix functionality using a popular real dataset collection called the Deep Learning Matrix Collection (DLMC), which were used in recent studies [1, 2].</p>
<p>Performance benchmarks scripts for matrix-matrix and matrix-vector ops (dense-sparse, sparse-sparse, and compare to dense-dense) are implemented here.</p>
<ul>
<li>
<p><code>matmul_bench.py</code> with <code>--operation sparse@sparse|sparse@dense</code> is for Sparse matrix-matrix multiplication (SPMM) performance test. It can run in forward and backward mode with <code>--backward-test</code>, on CPU or CUDA with <code>--with-cuda</code>, using different datasets from the dataset collection DLMC. For more details see <code>test.sh</code> file.</p>
</li>
<li>
<p><code>matmul_bench.py</code> with <code>--operation sparse@vector</code> is for Sparse matrix-vector multiplication (SPMV) performance test.</p>
</li>
</ul>
<p>References:</p>
<ol>
<li>
<p>Trevor Gale, Matei Zaharia, Cliff Young, Erich Elsen. Sparse GPU Kernels for Deep Learning. Proceedings of the International Conference for High Performance Computing, 2020. https://github.com/google-research/google-research/tree/master/sgk</p>
</li>
<li>
<p>Trevor Gale, Erich Elsen, Sara Hooker. The State of Sparsity in Deep Neural Networks. https://github.com/google-research/google-research/tree/master/state_of_sparsity</p>
</li>
</ol>