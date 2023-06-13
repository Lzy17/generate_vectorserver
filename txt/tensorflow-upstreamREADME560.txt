<h1>TFLite Serialization Tool</h1>
<p><strong>NOTE:</strong> This tool is intended for advanced users only, and should be used with
care.</p>
<p>The (C++) serialization library generates and writes a TFLite flatbuffer given
an <code>Interpreter</code> or <code>Subgraph</code>. Example use-cases include authoring models with
the <code>Interpreter</code> API, or updating models on-device (by modifying <code>tensor.data</code>
for relevant tensors).</p>
<h2>Serialization</h2>
<h3>Writing flatbuffer to file</h3>
<p>To write a TFLite model from an <code>Interpreter</code> (see <code>lite/interpreter.h</code>):
<code>std::unique_ptr&lt;tflite::Interpreter&gt; interpreter; // ...build/modify
interpreter... tflite::ModelWriter writer(interpreter.get()); std::string
filename = "/tmp/model.tflite"; writer.Write(filename);</code></p>
<p>Note that the above API does not support custom I/O tensors or custom ops yet.
However, it does support model with Control Flow.</p>
<p>To generate/write a flatbuffer for a particular <code>Subgraph</code> (see
<code>lite/core/subgraph.h</code>) you can use <code>SubgraphWriter</code>.</p>
<p><code>std::unique_ptr&lt;tflite::Interpreter&gt; interpreter;
// ...build/modify interpreter...
// The number of subgraphs can be obtained by:
// const int num_subgraphs = interpreter_-&gt;subgraphs_size();
// Note that 0 &lt;= subgraph_index &lt; num_subgraphs
tflite::SubgraphWriter writer(&amp;interpreter-&gt;subgraph(subgraph_index));
std::string filename = "/tmp/model.tflite";
writer.Write(filename);</code></p>
<p><code>SubgraphWriter</code> supports custom ops and/or custom I/O tensors.</p>
<h3>Generating flatbuffer in-memory</h3>
<p>Both <code>ModelWriter</code> and <code>SubgraphWriter</code> support a <code>GetBuffer</code> method to return
the generated flatbuffer in-memory:</p>
<p><code>std::unique_ptr&lt;uint8_t[]&gt; output_buffer;
size_t output_buffer_size;
tflite::ModelWriter writer(interpreter.get());
writer.GetBuffer(&amp;output_buffer, &amp;output_buffer_size);</code></p>
<h2>De-serialization</h2>
<p>The flatbuffers written as above can be de-serialized just like any other TFLite
model, for eg:</p>
<p><code>std::unique_ptr&lt;FlatBufferModel&gt; model =
    FlatBufferModel::BuildFromFile(filename);
tflite::ops::builtin::BuiltinOpResolver resolver;
InterpreterBuilder builder(*model, resolver);
std::unique_ptr&lt;Interpreter&gt; new_interpreter;
builder(&amp;new_interpreter);</code></p>