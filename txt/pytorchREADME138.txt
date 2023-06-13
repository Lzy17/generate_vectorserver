<h1>C++ Frontend Tests</h1>
<p>In this folder live the tests for PyTorch's C++ Frontend. They use the
<a href="https://github.com/google/googletest">GoogleTest</a> test framework.</p>
<h2>CUDA Tests</h2>
<p>To make a test runnable only on platforms with CUDA, you should suffix your
test with <code>_CUDA</code>, e.g.</p>
<p><code>cpp
TEST(MyTestSuite, MyTestCase_CUDA) { }</code></p>
<p>To make it runnable only on platforms with at least two CUDA machines, suffix
it with <code>_MultiCUDA</code> instead of <code>_CUDA</code>, e.g.</p>
<p><code>cpp
TEST(MyTestSuite, MyTestCase_MultiCUDA) { }</code></p>
<p>There is logic in <code>main.cpp</code> that detects the availability and number of CUDA
devices and supplies the appropriate negative filters to GoogleTest.</p>
<h2>Integration Tests</h2>
<p>Integration tests use the MNIST dataset. You must download it by running the
following command from the PyTorch root folder:</p>
<p><code>sh
$ python tools/download_mnist.py -d test/cpp/api/mnist</code></p>
<p>The required paths will be referenced as <code>test/cpp/api/mnist/...</code> in the test
code, so you <em>must</em> run the integration tests from the PyTorch root folder.</p>