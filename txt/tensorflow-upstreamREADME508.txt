<h1>TensorFlow</h1>
<p>TensorFlow is a computational dataflow graph library.</p>
<h2>Getting started</h2>
<h3>Python API example</h3>
<p>The following is an example python code to do a simple matrix multiply
of two constants and get the result from a locally-running TensorFlow
process.</p>
<p>First, bring in tensorflow python dependency</p>
<p>//third_party/py/tensorflow</p>
<p>to get the python TensorFlow API.</p>
<p>Then:</p>
<p>```python
import tensorflow as tf</p>
<p>with tf.Session():
  input1 = tf.constant(1.0, shape=[1, 1], name="input1")
  input2 = tf.constant(2.0, shape=[1, 1], name="input2")
  output = tf.matmul(input1, input2)</p>
<p># Run graph and fetch the output
  result = output.eval()
  print result
```</p>
<h3>C++ API Example</h3>
<p>If you are running TensorFlow locally, link your binary with</p>
<p>//third_party/tensorflow/core</p>
<p>and link in the operation implementations you want to supported, e.g.,</p>
<p>//third_party/tensorflow/core:kernels</p>
<p>An example program to take a GraphDef and run it using TensorFlow
using the C++ Session API:</p>
<p>```c++</p>
<h1>include <memory></h1>
<h1>include <string></h1>
<h1>include <vector></h1>
<h1>include "tensorflow/core/framework/graph.pb.h"</h1>
<h1>include "tensorflow/core/public/session.h"</h1>
<h1>include "tensorflow/core/framework/tensor.h"</h1>
<p>int main(int argc, char** argv) {
  // Construct your graph.
  tensorflow::GraphDef graph = ...;</p>
<p>// Create a Session running TensorFlow locally in process.
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession({}));</p>
<p>// Initialize the session with the graph.
  tensorflow::Status s = session-&gt;Create(graph);
  if (!s.ok()) { ... }</p>
<p>// Specify the 'feeds' of your network if needed.
  std::vector<std::pair\<string, tensorflow::Tensor>> inputs;</p>
<p>// Run the session, asking for the first output of "my_output".
  std::vector<tensorflow::Tensor> outputs;
  s = session-&gt;Run(inputs, {"my_output:0"}, {}, &amp;outputs);
  if (!s.ok()) { ... }</p>
<p>// Do something with your outputs
  auto output_vector = outputs[0].vec<float>();
  if (output_vector(0) &gt; 0.5) { ... }</p>
<p>// Close the session.
  session-&gt;Close();</p>
<p>return 0;
}
```</p>
<p>For a more fully-featured C++ example, see
<code>tensorflow/cc/tutorials/example_trainer.cc</code></p>