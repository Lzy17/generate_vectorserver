<h1>TensorFlow SavedModel</h1>
<p>[TOC]</p>
<h2>Overview</h2>
<p>SavedModel is the universal serialization format for
<a href="https://www.tensorflow.org/">TensorFlow</a> models.</p>
<p>SavedModel provides a language-neutral format to save machine-learning models
that is recoverable and hermetic. It enables higher-level systems and tools to
produce, consume and transform TensorFlow models.</p>
<h2>Guides</h2>
<ul>
<li><a href="https://www.tensorflow.org/guide/saved_model">Using the SavedModel Format</a></li>
<li><a href="https://www.tensorflow.org/guide/keras/save_and_serialize">Save and load Keras models</a></li>
<li><a href="https://www.tensorflow.org/tutorials/keras/save_and_load">Save and load with checkpointing in Keras</a></li>
<li><a href="https://www.tensorflow.org/guide/checkpoint">Training checkpoints</a></li>
<li><a href="https://www.tensorflow.org/tutorials/distribute/save_and_load">Save and load a model using a distribution strategy</a></li>
</ul>
<h2><a href="https://www.tensorflow.org/api_docs/python/tf/saved_model">Public API</a></h2>
<ul>
<li><a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/save"><code>tf.saved_model.save</code></a></li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/load"><code>tf.saved_model.load</code></a></li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/SaveOptions"><code>tf.saved_model.SaveOptions</code></a></li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/LoadOptions"><code>tf.saved_model.LoadOptions</code></a></li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/Asset"><code>tf.saved_model.Asset</code></a></li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf/saved_model/contains_saved_model"><code>tf.saved_model.contains_saved_model</code></a></li>
</ul>
<h3>Related Modules and Functions</h3>
<ul>
<li><a href="https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model"><code>tf.keras.models.save_model</code></a></li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model"><code>tf.keras.models.load_model</code></a></li>
<li><a href="https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint"><code>tf.train.Checkpoint</code></a></li>
</ul>
<h2>The SavedModel Format</h2>
<p>A SavedModel directory has the following structure:</p>
<p><code>assets/
assets.extra/
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb</code></p>
<ul>
<li>SavedModel protocol buffer<ul>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/saved_model.proto"><code>saved_model.pb</code></a>
    or <code>saved_model.pbtxt</code></li>
<li>Includes the graph definitions as <code>MetaGraphDef</code> protocol buffers.</li>
</ul>
</li>
<li>Assets<ul>
<li>Subfolder called <code>assets</code>.</li>
<li>Contains auxiliary files such as vocabularies, etc.</li>
</ul>
</li>
<li>Extra assets<ul>
<li>Subfolder where higher-level libraries and users can add their own
    assets that co-exist with the model, but are not loaded by the graph.</li>
<li>This subfolder is not managed by the SavedModel libraries.</li>
</ul>
</li>
<li>Variables<ul>
<li>Subfolder called <code>variables</code>.<ul>
<li><code>variables.data-?????-of-?????</code></li>
<li><code>variables.index</code></li>
</ul>
</li>
</ul>
</li>
</ul>
<hr />
<h2>SavedModel in TensorFlow 1.x</h2>
<p>SavedModel had slightly different semantics in TF 1.x. Conventions that are
generally only supported in TF 1.x are noted as such.</p>
<h3>Features</h3>
<p>The following is a summary of the features in SavedModel:</p>
<ul>
<li>(TF1-only) Multiple graphs sharing a single set of variables and assets can be added to a
  single SavedModel. Each graph is associated with a specific set of tags to
  allow identification during a load or restore operation.</li>
<li>(TF1-only) Support for <code>SignatureDefs</code><ul>
<li>Graphs that are used for inference tasks typically have a set of inputs
  and outputs. This is called a <code>Signature</code>.</li>
<li>SavedModel uses <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto">SignatureDefs</a>
  to allow generic support for signatures that may need to be saved with the graphs.</li>
<li>For commonly used SignatureDefs in the context of TensorFlow Serving,
  please see documentation <a href="https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/signature_defs.md">here</a>.</li>
</ul>
</li>
<li>Support for <code>Assets</code>.<ul>
<li>For cases where ops depend on external files for initialization, such as
  vocabularies, SavedModel supports this via <code>assets</code>.</li>
<li>Assets are copied to the SavedModel location and can be read when loading
  a specific meta graph def.</li>
</ul>
</li>
<li>Support to clear devices before generating the SavedModel.</li>
</ul>
<p>The following is a summary of features that are NOT supported in SavedModel.
Higher-level frameworks and tools that use SavedModel may provide these.</p>
<ul>
<li>Implicit versioning.</li>
<li>Garbage collection.</li>
<li>Atomic writes to the SavedModel location.</li>
</ul>
<h3>TF1 SavedModel Background</h3>
<p>SavedModel manages and builds upon existing TensorFlow primitives such as
<code>TensorFlow Saver</code> and <code>MetaGraphDef</code>. Specifically, SavedModel wraps a <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/training/saver.py">TensorFlow Saver</a>.
The Saver is primarily used to generate the variable checkpoints. SavedModel
will replace the existing <a href="https://github.com/tensorflow/tensorflow/tree/r1.15/tensorflow/contrib/session_bundle#tensorflow-inference-model-format">TensorFlow Inference Model Format</a>
as the canonical way to export TensorFlow graphs for serving.</p>
<h3>APIs</h3>
<p>The APIs for building and loading a SavedModel are described in this section.</p>
<h4>(TF1-only) Builder</h4>
<p>The SavedModel <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder.py">builder</a>
is implemented in Python.</p>
<p>The <code>SavedModelBuilder</code> class provides functionality to save multiple meta graph
defs, associated variables and assets.</p>
<p>To build a SavedModel, the first meta graph must be saved with variables.
Subsequent meta graphs will simply be saved with their graph definitions. If
assets need to be saved and written or copied to disk, they can be provided
when the meta graph def is added. If multiple meta graph defs are associated
with an asset of the same name, only the first version is retained.</p>
<h4>(TF1-only) Tags</h4>
<p>Each meta graph added to the SavedModel must be annotated with user specified
tags, which reflect the meta graph capabilities or use-cases.
More specifically, these tags typically annotate a meta graph with its
functionality (e.g. serving or training), and possibly hardware specific aspects
such as GPU.
In the SavedModel, the meta graph def whose tag-set exactly matches those
specified in the loader API, will be the one loaded by the loader.
If no meta graph def is found matching the specified tags, an error is returned.
For example, a loader with a requirement to serve on GPU hardware would be able
to load only meta graph annotated with tags='serve,gpu' by specifying this set
of tags in tensorflow::LoadSavedModel(...).</p>
<h4>Usage</h4>
<p>The typical usage of <code>builder</code> is as follows:</p>
<p>~~~python
export_dir = ...
...
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.TRAINING],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets)
...
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph(["bar-tag", "baz-tag"])
...
builder.save()
~~~</p>
<h4>(TF1-only) Stripping Default valued attributes</h4>
<p>The SavedModelBuilder class allows users to control whether default-valued
attributes must be stripped from the NodeDefs while adding a meta graph to the
SavedModel bundle. Both <code>SavedModelBuilder.add_meta_graph_and_variables</code> and
<code>SavedModelBuilder.add_meta_graph</code> methods accept a Boolean flag
<code>strip_default_attrs</code> that controls this behavior.</p>
<p>If <code>strip_default_attrs</code> is <code>False</code>, the exported MetaGraphDef will have the
default valued attributes in all it's NodeDef instances. This can break forward
compatibility with a sequence of events such as the following:</p>
<ul>
<li>An existing Op (<code>Foo</code>) is updated to include a new attribute (<code>T</code>) with a
  default (<code>bool</code>) at version 101.</li>
<li>A model producer (such as a Trainer) binary picks up this change
  (version 101) to the OpDef and re-exports an existing model that uses Op <code>Foo</code>.</li>
<li>A model consumer (such as Tensorflow Serving) running an older binary
  (version 100) doesn't have attribute <code>T</code> for Op <code>Foo</code>, but tries to import
  this model. The model consumer doesn't recognize attribute <code>T</code> in a NodeDef
  that uses Op <code>Foo</code> and therefore fails to load the model.</li>
</ul>
<p>By setting <code>strip_default_attrs</code> to <code>True</code>, the model producers can strip away
any default valued attributes in the NodeDefs. This helps ensure that newly
added attributes with defaults don't cause older model consumers to fail loading
models regenerated with newer training binaries.</p>
<p>TIP: If you care about forward compatibility, then set <code>strip_default_attrs</code>
to <code>True</code> while using <code>SavedModelBuilder.add_meta_graph_and_variables</code> and
<code>SavedModelBuilder.add_meta_graph</code>.</p>
<h3>Loader</h3>
<p>The SavedModel loader is implemented in C++ and Python.</p>
<h4>(TF1-only) Python</h4>
<p>The Python version of the SavedModel <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/loader.py">loader</a>
provides load and restore capability for a SavedModel. The <code>load</code> operation
requires the session in which to restore the graph definition and variables, the
tags used to identify the meta graph def to load and the location of the
SavedModel. Upon a load, the subset of variables and assets supplied as part of
the specific meta graph def, will be restored into the supplied session.</p>
<p>~~~python
export_dir = ...
...
with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)
  ...
~~~</p>
<h4>C++</h4>
<p>The C++ version of the SavedModel <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/loader.h">loader</a>
provides an API to load a SavedModel from a path, while allowing
<code>SessionOptions</code> and <code>RunOptions</code>. Similar to the Python version, the C++
version requires the tags associated with the graph to be loaded, to be
specified. The loaded version of SavedModel is referred to as <code>SavedModelBundle</code>
and contains the meta graph def and the session within which it is loaded.</p>
<p>~~~c++
const string export_dir = ...
SavedModelBundle bundle;
...
LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagTrain},
               &amp;bundle);
~~~</p>
<h3>Constants</h3>
<p>SavedModel offers the flexibility to build and load TensorFlow graphs for a
variety of use-cases. For the set of most common expected use-cases,
SavedModel's APIs provide a set of constants in Python and C++ that are easy to
reuse and share across tools consistently.</p>
<h4>(TF1-specific) Tag constants</h4>
<p>Sets of tags can be used to uniquely identify a <code>MetaGraphDef</code> saved in a
SavedModel. A subset of commonly used tags is specified in:</p>
<ul>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/tag_constants.py">Python</a></li>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/tag_constants.h">C++</a>.</li>
</ul>
<h4>(TF1-specific) Signature constants</h4>
<p>SignatureDefs are used to define the signature of a computation supported in a
TensorFlow graph. Commonly used input keys, output keys and method names are
defined in:</p>
<ul>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py">Python</a></li>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/signature_constants.h">C++</a>.</li>
</ul>