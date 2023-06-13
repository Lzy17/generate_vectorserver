<h1>C++ gradients</h1>
<p>Gradients are currently being ported from
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops">python</a>
to C++ (in this directory).</p>
<p>Contributions are welcome and much appreciated; please follow the instructions
below.</p>
<ol>
<li>
<p>Create the op gradient function in <code>foo_grad.cc</code> corresponding to the
    <code>foo_grad.py</code> file where the op originated (i.e. <code>array_grad.py</code> op
    gradients should be written in <code>array_grad.cc</code>).</p>
</li>
<li>
<p>Write the op gradient with the following naming scheme:</p>
<p><code>Status OpNameGrad(const Scope&amp; scope, const Operation&amp; op,
                  const std::vector&lt;Output&gt;&amp; grad_inputs,
                  std::vector&lt;Output&gt;* grad_outputs) {
  ...
  return scope.status();
}
REGISTER_GRADIENT_OP("OpName", OpNameGrad);</code></p>
</li>
<li>
<p>Ops gradients are implemented by using the
    <a href="https://www.tensorflow.org/api_docs/cc/">C++ API</a>.</p>
</li>
<li>
<p>Tests should be included in <code>foo_grad_test.cc</code>. Please see
    <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/gradients/array_grad_test.cc"><code>array_grad_test.cc</code></a>
    for many examples. Tests are as simple as, creating a placeholder input for
    the op's inputs and calling <code>RunTest</code> (<code>RunTest</code> uses a
    <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/framework/gradient_checker.cc">gradient checker</a>
    to verify that the theoretical gradient matches the numeric gradient). For
    example:</p>
<p><code>TEST_F(ArrayGradTest, IdentityGrad) {
  TensorShape shape({5, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Identity(scope_, x);
  RunTest(x, shape, y, shape);
}</code></p>
</li>
</ol>
<p>NOTE: There are some ops that require features from the C++ API that are not yet
implemented.</p>
<ul>
<li>
<p>Ops that require PartialTensorShape information cannot yet be implemented.</p>
</li>
<li>
<p>Ops that require SparseTensor or IndexSlices (currently only in python)
    cannot yet be implemented.</p>
</li>
<li>
<p>Maybe more.</p>
</li>
</ul>
<p>For questions: Please create an issue assigned to suharshs.</p>