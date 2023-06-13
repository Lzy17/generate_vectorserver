<h1>Keras SavedModel</h1>
<p>For questions, feedback, and feature requests please file a bug/contact kathywu@</p>
<h2>TensorFlow Core SavedModel implementation</h2>
<p>In TensorFlow 2.0, all saving and loading implementations revolve around the
object graph generated from a root trackable object, and all trackable objects
connected to it through attributes. Program building blocks such as variables,
assets, and tables, and high level objects like Optimizers and Layers all
subclass the trackable class. Other objects like TensorFlow functions and
concrete functions are also saved as nodes in the object graph. When loading a
SavedModel, the object graph is used to recreate the structure of the original
object.</p>
<p>Please see the links below for more details:</p>
<ul>
<li><a href="https://www.tensorflow.org/guide/saved_model">Saved Model Guide</a></li>
<li><a href="https://www.tensorflow.org/guide/checkpoint">Checkpoint Guide</a></li>
</ul>
<h2>Keras SavedModel implementation</h2>
<h3>Overview</h3>
<p>Keras object serialization is built on top of the core serialization.</p>
<p>All attributes that impact model execution or inspection are saved to the
SavedModel to allow the model to be recreated. These attributes are divided into
three categories:</p>
<ol>
<li>python properties (e.g., layer name, layer config)</li>
<li>objects (e.g. data structures like list of variables or layers)</li>
<li>functions (e.g. call function, loss functions)</li>
</ol>
<p>Trackable objects and TensorFlow functions are represented as nodes in the
trackable object graph, and each node in the graph stores information about
their python properties.</p>
<p>Since many attributes in Keras Layers/Models are not Trackable objects or
tf.functions, these attributes are wrapped as trackable objects/tf.functions at
serialization time. For example, <code>layer.variables</code> is implemented as a python
property that appends the lists of trainable/nontrainable variables. During
serialization, a new Trackable List object is created and saved to the
<code>variables</code> attribute. Another example is the call function. Most models do not
decorate their call function with <code>tf.function</code>, since Keras will take care of
the graph/function management. When the model is saved, the call function is
wrapped in a <code>tf.function</code> and added to the <code>__call__</code> attribute.</p>
<h3><code>keras_api</code> attribute</h3>
<p>Many attributes are only relevant for revivability. Instead of attaching these
directly to the exported object, they are saved to a new <code>keras_api</code> trackable
object that is then attached to the exported object. This avoids cluttering the
exported object with objects/functions that are only used by the Keras library.</p>
<p>For example, <code>__call__</code> and <code>call_and_return_conditional_losses</code> are functions
saved for all models. The <code>__call__</code> function is attached directly to the
exported object, while <code>call_and_return_conditional_losses</code> is attached to a
separate object. Say a user saves the model, then loads the SavedModel using the
core loader (tf.saved_model.load which does not rely on the Keras library to
revive the model).</p>
<p>The loaded object will have a structure that looks like:</p>
<p><code>loaded object -- __call__
                -- keras_api -- __call__
                             -- call_and_return_conditional_losses</code></p>
<p>The two call functions may be accessed through:</p>
<ul>
<li><code>loaded.__call__</code> or <code>loaded.keras_api.__call__</code></li>
<li><code>loaded.keras_api.call_and_return_conditional_losses</code>.</li>
</ul>
<h3>Saving details</h3>
<p>Keras Layers use a helper abstract class and an attribute validator class to
define and standardize the serialization implementation:</p>
<ul>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/saved_model/base_serialization.py"><code>SerializationImpl</code></a>:
Ensures that layer python properties are saved as a serialized JSON string in
the metadata field, and gathers all attributes to save with the Keras object.
Please see the docstrings in each of the abstract methods/properties to see what
is required.</li>
<li><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/saved_model/serialized_attributes.py?"><code>SerializedAttributes</code></a>:
Tracks all of the attributes that must be saved with a Keras object. Objects and
functions may be specified to be "keras_only", meaning that they will only
appear in the <code>keras_api</code> attribute.</li>
</ul>
<p>The base <code>Layer</code> serialization is defined in
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/saved_model/layer_serialization.py"><code>layer_serialization.py</code></a>.
See <code>LayerAttributes</code> and <code>LayerSerializationImpl</code>.</p>
<p><strong>Adding a new attribute to base Layer SavedModel</strong></p>
<ol>
<li>Add a new attributes to <code>LayerAttributes</code>.</li>
<li>Modify <code>LayerSerializationImpl</code> internal methods:</li>
</ol>
<p>a. If adding a python property, add the key-value item to the dictionary
   returned by <code>_python_properties_internal</code></p>
<p>b.If adding a new object/function, modify the dictionary returned by
   <code>_get_serialized_attributes_internal</code>.</p>
<p><strong>Adding custom serialization for a Layer subclass.</strong></p>
<ol>
<li>Create a new attribute validator by copying <code>LayerAttributes</code>, and add any
new attributes to serialize.</li>
<li>Subclass <code>LayerSerializationImpl</code></li>
<li>Implement <code>_python_properties_internal</code> and/or
<code>_get_serialized_attributes_internal</code> to return the new attributes.</li>
</ol>
<p>Unless you are modifying the loader (see section below on loading), please keep
the <code>object_identifier</code> the same.</p>
<p>These instructions also carry over for modifying
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/saved_model/model_serialization.py">Model</a>
and
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/saved_model/network_serialization.py">Network</a>
serialization.</p>
<h3>Loading details</h3>
<p>TODO(kathywu): Will write this section when the loading code is moved into
*_serialization.py files.</p>