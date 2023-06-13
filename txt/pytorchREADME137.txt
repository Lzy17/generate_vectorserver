<h2>Run distributed tensor tests:</h2>
<p>from root, run (either CPU or GPU)</p>
<p><code>pytest test/spmd/tensor/test_tensor.py</code></p>
<p><code>pytest test/spmd/tensor/test_ddp.py</code></p>
<p>run specific test case and print stdout/stderr:</p>
<p><code>pytest test/spmd/tensor/test_tensor.py -s -k test_tensor_from_local</code></p>