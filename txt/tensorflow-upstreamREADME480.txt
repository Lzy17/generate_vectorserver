<h1>Tensorflow Distribute Libraries</h1>
<h2>Overview</h2>
<p>tf.distribute.Strategy is a TensorFlow API to distribute training across
multiple GPUs, multiple machines or TPUs. Using this API, users can distribute
their existing models and training code with minimal code changes.</p>
<p>It can be used with TensorFlow's high level APIs, tf.keras and tf.estimator,
with just a couple of lines of code change. It does so by changing the
underlying components of TensorFlow to become strategy-aware.
This includes variables, layers, models, optimizers, metrics, summaries,
and checkpoints.</p>
<h2>Documentation</h2>
<p><a href="https://www.tensorflow.org/guide/distributed_training">Distributed Training Guide</a></p>
<p><a href="https://www.tensorflow.org/tutorials/distribute/keras">Distributed Training With Keras Tutorial</a></p>
<p><a href="https://www.tensorflow.org/tutorials/distribute/custom_training">Distributed Training With Custom Training Loops Tutorial</a></p>
<p><a href="https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras">Multiworker Training With Keras Tutorial</a></p>
<p><a href="https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator">Multiworker Training With Estimator Tutorial</a></p>
<p><a href="https://www.tensorflow.org/tutorials/distribute/save_and_load">Save and Load with Distribution Strategy</a></p>
<h2>Simple Examples</h2>
<h3>Using compile fit with GPUs.</h3>
<p>```python</p>
<h1>Create the strategy instance. It will automatically detect all the GPUs.</h1>
<p>mirrored_strategy = tf.distribute.MirroredStrategy()</p>
<h1>Create and compile the keras model under strategy.scope()</h1>
<p>with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  model.compile(loss='mse', optimizer='sgd')</p>
<h1>Call model.fit and model.evaluate as before.</h1>
<p>dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
model.fit(dataset, epochs=2)
model.evaluate(dataset)
```</p>
<h3>Custom training loop with TPUs.</h3>
<p>```python</p>
<h1>Create the strategy instance.</h1>
<p>tpu_strategy = tf.distribute.TPUStrategy(resolver)</p>
<h1>Create the keras model under strategy.scope()</h1>
<p>with tpu_strategy.scope():
  model = keras.layers.Dense(1, name="dense")</p>
<h1>Create custom training loop body as tf.function.</h1>
<p>@tf.function
def train_step(iterator):
  def step_fn(inputs):
    images, targets = inputs
    with tf.GradientTape() as tape:
      outputs = model(images)
      loss = tf.reduce_sum(outputs - targets)
    grads = tape.gradient(loss, model.variables)
    return grads</p>
<p>return tpu_strategy.run(
      step_fn, args=(next(iterator),))</p>
<h1>Run the loop body once on at dataset.</h1>
<p>dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10
input_iterator = iter(tpu_strategy.experimental_distribute_dataset(dataset))
train_step(input_iterator)
```</p>
<h2>Testing</h2>
<p>Tests here should cover all distribution strategies to ensure feature parity.
This can be done using the test decorators in <code>strategy_combinations.py</code>.</p>