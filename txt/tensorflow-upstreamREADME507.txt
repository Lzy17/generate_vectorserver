<h1>Registrations</h1>
<p>To configure SaveModel or checkpointing beyond the basic saving and loading
steps [documentation TBD], registration is required.</p>
<p>Currently, only TensorFlow-internal
registrations are allowed, and must be added to the allowlist.</p>
<ul>
<li><code>tensorflow.python.saved_model.registration.register_tf_serializable</code></li>
<li>Allowlist: tf_serializable_allowlist.txt</li>
<li><code>tensorflow.python.saved_model.registration.register_tf_checkpoint_saver</code></li>
<li>Allowlist: tf_checkpoint_saver_allowlist.txt</li>
</ul>
<p>[TOC]</p>
<h2>SavedModel serializable registration</h2>
<p>Custom objects must be registered in order to get the correct deserialization
method when loading. The registered name of the class is saved to the proto.</p>
<p>Keras already has a similar mechanism for registering serializables:
<a href="https://www.tensorflow.org/api_docs/python/tf/keras/utils/register_keras_serializable"><code>tf.keras.utils.register_keras_serializable(package, name)</code></a>.
This has been imported to core TensorFlow:</p>
<p><code>python
registration.register_serializable(package, name)
registration.register_tf_serializable(name)  # If TensorFlow-internal.</code></p>
<ul>
<li>package: The package that this class belongs to.</li>
<li>name: The name of this class. The registered name that is saved in the proto
    is "{package}.{name}" (for TensorFlow internal registration, the package
    name is <code>tf</code>)</li>
</ul>
<h2>Checkpoint saver registration</h2>
<p>If <code>Trackables</code> share state or require complicated coordination between multiple
<code>Trackables</code> (e.g. <code>DTensor</code>), then users may register a save and restore
functions for these objects.</p>
<p><code>tf.saved_model.register_checkpoint_saver(
    predicate, save_fn=None, restore_fn=None):</code></p>
<ul>
<li><code>predicate</code>: A function that returns <code>True</code> if a <code>Trackable</code> object should
    be saved using the registered <code>save_fn</code> or <code>restore_fn</code>.</li>
<li><code>save_fn</code>: A python function or <code>tf.function</code> or <code>None</code>. If <code>None</code>, run the
    default saving process which calls <code>Trackable._serialize_to_tensors</code>.</li>
<li><code>restore_fn</code>: A <code>tf.function</code> or <code>None</code>. If <code>None</code>, run the default
    restoring process which calls <code>Trackable._restore_from_tensors</code>.</li>
</ul>
<p><strong><code>save_fn</code> details</strong></p>
<p><code>@tf.function  # optional decorator
def save_fn(trackables, file_prefix): -&gt; List[shard filenames]</code></p>
<ul>
<li><code>trackables</code>: A dictionary of <code>{object_prefix: Trackable}</code>. The
    object_prefix can be used as the object names, and uniquely identify each
    <code>Trackable</code>. <code>trackables</code> is the filtered set of trackables that pass the
    predicate.</li>
<li><code>file_prefix</code>: A string or string tensor of the checkpoint prefix.</li>
<li><code>shard filenames</code>: A list of filenames written using <code>io_ops.save_v2</code>, which
    will be merged into the checkpoint data files. These should be prefixed by
    <code>file_prefix</code>.</li>
</ul>
<p>This function can be a python function, in which case shard filenames can be an
empty list (if the values are written without the <code>SaveV2</code> op).</p>
<p>If this function is a <code>tf.function</code>, then the shards must be written using the
SaveV2 op. This guarantees the checkpoint format is compatible with existing
checkpoint readers and managers.</p>
<p><strong><code>restore_fn</code> details</strong></p>
<p><code>@tf.function  # required decorator
def restore_fn(trackables, file_prefix): -&gt; None</code></p>
<p>A <code>tf.function</code> with the spec:</p>
<ul>
<li><code>trackables</code>: A dictionary of <code>{object_prefix: Trackable}</code>. The
    <code>object_prefix</code> can be used as the object name, and uniquely identifies each
    Trackable. The Trackable objects are the filtered results of the registered
    predicate.</li>
<li><code>file_prefix</code>: A string or string tensor of the checkpoint prefix.</li>
</ul>
<p><strong>Why are restore functions required to be a <code>tf.function</code>?</strong> The short answer
is, the SavedModel format must maintain the invariant that SavedModel packages
can be used for inference on any platform and language. SavedModel inference
needs to be able to restore checkpointed values, so the restore function must be
directly encoded into the SavedModel in the Graph. We also have security
measures over FunctionDef and GraphDef, so users can check that the SavedModel
will not run arbitrary code (a feature of <code>saved_model_cli</code>).</p>
<h2>Example</h2>
<p>Below shows a <code>Stack</code> module that contains multiple <code>Parts</code> (a subclass of
<code>tf.Variable</code>). When a <code>Stack</code> is saved to a checkpoint, the <code>Parts</code> are stacked
together and a single entry in the checkpoint is created. The checkpoint value
is restored to all of the <code>Parts</code> in the <code>Stack</code>.</p>
<p>```
@registration.register_serializable()
class Part(resource_variable_ops.ResourceVariable):</p>
<p>def <strong>init</strong>(self, value):
    self._init_from_args(value)</p>
<p>@classmethod
  def _deserialize_from_proto(cls, **kwargs):
    return cls([0, 0])</p>
<p>@registration.register_serializable()
class Stack(tracking.AutoTrackable):</p>
<p>def <strong>init</strong>(self, parts=None):
    self.parts = parts</p>
<p>@def_function.function(input_signature=[])
  def value(self):
    return array_ops_stack.stack(self.parts)</p>
<p>def get_tensor_slices(trackables):
  tensor_names = []
  shapes_and_slices = []
  tensors = []
  restored_trackables = []
  for obj_prefix, obj in trackables.items():
    if isinstance(obj, Part):
      continue  # only save stacks
    tensor_names.append(obj_prefix + "/value")
    shapes_and_slices.append("")
    x = obj.value()
    with ops.device("/device:CPU:0"):
      tensors.append(array_ops.identity(x))
    restored_trackables.append(obj)</p>
<p>return tensor_names, shapes_and_slices, tensors, restored_trackables</p>
<p>def save_stacks_and_parts(trackables, file_prefix):
  """Save stack and part objects to a checkpoint shard."""
  tensor_names, shapes_and_slices, tensors, _ = get_tensor_slices(trackables)
  io_ops.save_v2(file_prefix, tensor_names, shapes_and_slices, tensors)
  return file_prefix</p>
<p>def restore_stacks_and_parts(trackables, merged_prefix):
  tensor_names, shapes_and_slices, tensors, restored_trackables = (
      get_tensor_slices(trackables))
  dtypes = [t.dtype for t in tensors]
  restored_tensors = io_ops.restore_v2(merged_prefix, tensor_names,
                                       shapes_and_slices, dtypes)
  for trackable, restored_tensor in zip(restored_trackables, restored_tensors):
    expected_shape = trackable.value().get_shape()
    restored_tensor = array_ops.reshape(restored_tensor, expected_shape)
    parts = array_ops_stack.unstack(restored_tensor)
    for part, restored_part in zip(trackable.parts, parts):
      part.assign(restored_part)</p>
<p>registration.register_checkpoint_saver(
    name="stacks",
    predicate=lambda x: isinstance(x, (Stack, Part)),
    save_fn=save_stacks_and_parts,
    restore_fn=restore_stacks_and_parts)
```</p>