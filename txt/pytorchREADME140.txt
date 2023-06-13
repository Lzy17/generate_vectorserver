<h1>JIT C++ Tests</h1>
<h2>Adding a new test</h2>
<p>First, create a new test file. Test files should have be placed in this
directory, with a name that starts with <code>test_</code>, like <code>test_foo.cpp</code>.</p>
<p>In general a single test suite</p>
<p>Add your test file to the <code>JIT_TEST_SRCS</code> list in <code>test/cpp/jit/CMakeLists.txt</code>.</p>
<p>A test file may look like:
```cpp</p>
<h1>include <gtest/gtest.h></h1>
<p>using namespace ::torch::jit</p>
<p>TEST(FooTest, BarBaz) {
   // ...
}</p>
<p>// Append '_CUDA' to the test case name will automatically filter it out if CUDA
// is not compiled.
TEST(FooTest, NeedsAGpu_CUDA) {
   // ...
}</p>
<p>// Similarly, if only one GPU is detected, tests with <code>_MultiCUDA</code> at the end
// will not be run.
TEST(FooTest, NeedsMultipleGpus_MultiCUDA) {
   // ...
}
```</p>
<h2>Building and running the tests</h2>
<p>The following commands assume you are in PyTorch root.</p>
<p>```bash</p>
<h1>... Build PyTorch from source, e.g.</h1>
<p>python setup.py develop</p>
<h1>(re)build just the binary</h1>
<p>ninja -C build bin/test_jit</p>
<h1>run tests</h1>
<p>build/bin/test_jit --gtest_filter='glob_style_filter*'
```</p>