<h1>nomnigraph</h1>
<p>nomnigraph is caffe2's graph transformation subsystem</p>
<h2>Usage</h2>
<p>The output of <code>caffe2::convertToNNModule(caffe2::NetDef)</code> (found in <code>caffe2/opt</code>) is an <code>NNModule</code>.
The output of <code>caffe2::convertToCaffe2Proto(nom::repr::NNModule*, caffe2::NetDef)</code> is a <code>NetDef</code>.
<code>convertToCaffe2Proto(convertToNNModule(n), n)</code> should basically return an unchanged network.</p>
<p>An <code>NNModule</code> is composed of both <code>dataFlow</code> and <code>controlFlow</code> graphs.</p>
<p>Creating a new operator is straightforward.
<code>cpp
auto reluNode = nn.dataFlow.createNode(make_unique&lt;nom::repr::Relu&gt;());</code>
The line above does a few things worth talking about.</p>
<p>1) It creates a new node using the graph API (both dataFlow and controlFlow are <code>Graph</code>s).
2) It instantiates the node with data, specifically a <code>unique_ptr</code> to a neural network operator.
3) This <code>unique_ptr</code> contains a type that inherits from <code>NeuralNetOperator</code> and forms the fundamental representation described in the IR section below.</p>
<p>Inserting this operator into the graph would look something like this:</p>
<p><code>cpp
auto edge = nn.dataFlow.createEdge(convOutputTensorNode, reluNode);</code></p>
<p>Some notes here:
1) Again the graph API is used to insert the node into the graph with an edge.
2) Operators are strictly connected to Tensors, not other operators.</p>
<h2>IR</h2>
<p>nomnigraph has a <em>parallel</em> representation that can contain annotations with caffe2's OperatorDef.</p>
<p>If you call <code>caffe2::convertToNNModule(caffe2::NetDef)</code>, every operator in the <code>NNModule</code> will be annotated with a reference to the original operator in the net.</p>
<p>This means you should not delete the original protobuf.</p>
<p><code>cpp
auto conv = repr::nn::get&lt;repr::Conv&gt;(convNode);
if (conv-&gt;getAnnotation()) {
  auto annotation = dyn_cast&lt;caffe2::Caffe2Annotation&gt;(conv-&gt;getMutableAnnotation());
  OperatorDef* op = annotation-&gt;getMutableOperatorDef();
  // Do stuff with the caffe2 protobuf
}</code></p>
<p>If you create a new op, as shown in the example above and copied here:
<code>cpp
auto reluNode = nn.dataFlow.createNode(make_unique&lt;nom::repr::Relu&gt;());</code>
it will not have a caffe2 annotation.</p>
<p>How does <code>caffe2::convertToCaffe2Proto(nom::repr::NNModule*, caffe2::NetDef)</code> deal with this?</p>
<p>Operators are either generated manually (see the implementation in <code>caffe2/opt/converter.cc</code>) or automatically.
The automatic generation is done by simply setting the operator <code>type</code> to the name of the operator.
If you'd like to add your own operator to a net and need it to be generated (i.e. are writing a transform that inserts
new nodes which have attributes likes args) you will need to add your own code to <code>caffe2/opt/converter.cc</code>.</p>
<p>Do not create <code>OperatorDef</code>s in the transformation itself! This is an anti-pattern as the logic becomes less portable.</p>
<h2>API</h2>
<p>Below is a subset of selected API calls that are quite useful.  Lower level manipulation calls are omitted.</p>
<h3>Graph transformation API</h3>
<p>Nomnigraph provides a ReplaceSubgraph API to perform graph transformation operations without having to write custom subgraph matching logic. The main header file is <a href="include/nomnigraph/Transformations/SubgraphMatcher.h">SubgraphMatcher.h</a>.</p>
<p>ReplaceSubgraph API takes in
- A subgraph pattern to be matched
- A graph to be scanned for matching patterns
- A ReplaceGraph lambda function that takes in a matched subgraph; callers should implement specific graph transformation operation in the lambda.</p>
<p>The ReplaceSubgraph implementation takes care of the pattern matching part and also provides tools for callers to implement graph transformation logic with less effort.</p>
<p>Example usage of the API can be found in <a href="tests/subgraph_matcher_test.cc">subgraph_matcher_test.cc</a></p>
<p>Example usage of the API for NNGraph can be found in <a href="tests/neural_net_test.cc">neural_net_test.cc</a></p>
<h3>Graph API</h3>
<p>Nomnigraph's core graph APIs provide a generic graph data structure and basic graph manipulation abilities. The main header file is <a href="include/nomnigraph/Graph/Graph.h">Graph.h</a>.</p>
<p>```cpp
auto g = Graph<T>(); // Constructor</p>
<p>Graph<T>::NodeRef n = g.createNode(T t); // Returns reference to the node</p>
<p>Graph<T>::EdgeRef e = g.createEdge(n1, n2); // Returns reference to the edge</p>
<p>g.deleteNode(n); // Deletes the node and all of its in/out edges from the graph
// Use g.deleteNode(n, false); to keep the edges around.</p>
<p>g.deleteEdge(e); // Deletes the edge between two nodes.</p>
<p>auto e = g.getEdge(n1, n2); // Gets the first edge that has n1 as a tail and n2 as the head.</p>
<p>auto ns = g.getMutableNodes(); // Returns a vector of Graph<T>::NodeRef</p>
<p>auto es = g.getMutableEdges(); // Returns a vector of Graph<T>::EdgeRef</p>
<p>T d = n-&gt;data(); // Get the data stored at the node
```</p>
<h3>NN API</h3>
<p>NN (NeuralNet) extends core Graph with functionalities specific to neural network computation graph. The main header file is <a href="include/nomnigraph/Representations/NeuralNet.h">NeuralNet.h</a>.</p>
<p>Type checking &amp; data accessing</p>
<p>```cpp
repr::NNModule nn = ...;
using namespace nom;</p>
<p>repr::NNGraph::NodeRef n;  // Canonical node of the neural network</p>
<p>bool b = repr::nn::is<repr::Tensor>(n); // Checks the type stored on the node.  (Works with parent types.)</p>
<p>repr::Conv* c = repr::nn::get<repr::Conv>(n); // Returns a pointer to the NeuralNetOperator or NeuralNetData in the node
```</p>
<p>Iterate through nodes in a NNGraph.
<code>cpp
auto pairs = dataIterator(nn); // A useful paradigm for iterating through nodes and corresponding data in no particular order.
auto nodeRefs = nodeIterator(nn); // Iterate through nodes in no particular order.
// See https://github.com/pytorch/pytorch/blob/main/caffe2/opt/mobile.cc#L106-L109</code></p>
<p>These functions make it easy to check attributes on nodes.
```cpp
// -- Tensor node functions --
bool b = hasProducer(tensorNode);  // Checks for producers.
auto n = getProducer(tensorNode); // Returns the producer of the tensor
bool b = hasConsumer(tensorNode); // Checks for consumers.
std::vector<NNGraph::NodeRef> consumers = getConsumers(tensorNode); // Returns a vector of all consumers of the tensor.</p>
<p>// -- Operator node functions --
bool b = hasInputs(n); // Checks if there are any input tensors.
std::vector<NNGraph::NodeRef> getInputs(n); // Returns a vector of all the input tensor nodes.
std::vector<NNGraph::NodeRef> getOutputs(n); // Returns a vector of all the output tensor nodes.
```</p>
<p>These functions are less commonly useful
```cpp
coalesceInsertedDataDependencies(&amp;nn); // Fixes up all the inserted dependencies in the dataflow graph.</p>
<p>insertOp<repr::Relu>(nn.dataFlow, n1, n2); // Inserts an operator into the dataflow graph and creates a new blob to do so.
// n1 or n2 must be a tensor and the inserted blob inherits the name from that, appending an underscore.</p>
<p>convertNode<repr::ConvRelu>(nn.dataFlow, n);  // Converts the data at the node to a new node by calling the passed in type with the old node's data as the constructor argument.
```</p>