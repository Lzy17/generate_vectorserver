<h1>TensorExpr C++ Tests</h1>
<h2>How to add a new test</h2>
<p>First, create a new test file. Test files should have be placed in this
directory, with a name that starts with <code>test_</code>, like <code>test_foo.cpp</code>.</p>
<p>Here is an example test file you can copy-paste.
```cpp</p>
<h1>include <test/cpp/tensorexpr/test_base.h></h1>
<p>// Tests go in torch::jit
namespace torch {
namespace jit {</p>
<p>// 1. Test cases are void() functions.
// 2. They start with the prefix <code>test</code>
void testCaseOne() {
    // ...
}</p>
<p>void testCaseTwo() {
    // ...
}
}
}
```</p>
<p>Then, register your test in <code>tests.h</code>:
<code>cpp
// Add to TH_FORALL_TESTS_CUDA instead for CUDA-requiring tests
#define TH_FORALL_TESTS(_)             \
  _(ADFormulas)                        \
  _(Attributes)                        \
  ...
  _(CaseOne)  // note that the `test` prefix is omitted.
  _(CaseTwo)</code></p>
<p>We glob all the test files together in <code>CMakeLists.txt</code> so that you don't
have to edit it every time you add a test. Unfortunately, this means that in
order to get the build to pick up your new test file, you need to re-run
cmake:
<code>python setup.py build --cmake</code></p>
<h2>How do I run the tests?</h2>
<p>The following commands assume you are in PyTorch root.</p>
<p><code>bash
 # (re)build the test binary
 ninja build/bin/test_tensorexpr
 # run
 build/bin/test_tensorexpr --gtest_filter='glob_style_filter*'</code></p>