<h1>TensorFlow Lite Converter</h1>
<p>The TensorFlow Lite Converter converts TensorFlow graphs into
TensorFlow Lite graphs. There are additional usages that are also detailed in
the usage documentation.</p>
<h2>Usage documentation</h2>
<p>Usage information is given in these documents:</p>
<ul>
<li><a href="../g3doc/r1/convert/cmdline_reference.md">Command-line glossary</a></li>
<li><a href="../g3doc/r1/convert/cmdline_examples.md">Command-line examples</a></li>
<li><a href="../g3doc/r1/convert/python_api.md">Python API examples</a></li>
</ul>
<h2>Where the converter fits in the TensorFlow landscape</h2>
<p>Once an application developer has a trained TensorFlow model, the TensorFlow
Lite Converter will accept
that model and generate a TensorFlow Lite
<a href="https://google.github.io/flatbuffers/">FlatBuffer</a> file. The converter currently supports
<a href="https://www.tensorflow.org/guide/saved_model#using_savedmodel_with_estimators">SavedModels</a>,
frozen graphs (models generated via
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py">freeze_graph.py</a>),
and <code>tf.Keras</code> model files.  The TensorFlow Lite FlatBuffer file can be shipped
to client devices, generally mobile devices, where the TensorFlow Lite
interpreter handles them on-device.  This flow is represented in the diagram
below.</p>
<p><img alt="drawing" src="../g3doc/r1/images/convert/workflow.svg" /></p>