<h1>Lightning callbacks for data sparsifier and scheduler</h1>
<p><strong>These are callback scripts for lightning and does not introduce pytorch lightning dependency on PyTorch.</strong></p>
<h2>Introduction</h2>
<p>Callbacks for PytorchLightning that specifies on when and how to to sparsify the data weights of the model.</p>
<h2>Types of Data Sparsity Callbacks</h2>
<p>There are 2 types of data sparsity callbacks
1. <strong>Post Training data sparsifier callback</strong>: Sparsification of the model parameters <em>post</em> training.</p>
<ol>
<li><strong>Training Aware data sparsifier callback</strong>: Sparsification of the model parameters <em>during</em> training.</li>
</ol>
<h2>API Design</h2>
<ol>
<li>
<p><code>PostTrainingDataSparsity</code>: callback class that sparsifies the model parameters post training. Accepts</p>
<ol>
<li><code>data_sparsifier_class</code>: class/type of data sparsifier that needs to be used. Only the class should be passed, the data sparsifier object
will be created internally and will be attached to the model by the callback whenever necessary.</li>
<li><code>data_sparsifier_args</code>: the arguments/config for the data sparsifier constructor that will be used while creating the object.</li>
</ol>
<p>Example:
<code>from data_sparsity import PostTrainingDataSparsity
sparsifier_args = {
    'sparsity_level': 0.5,
    'sparse_block_shape': (1, 4),
    'zeros_per_block': 4
}
pt_callback = PostTrainingDataSparsity(data_sparsifier_class=DataNormSparsifier, data_sparsifier_args=sparsifier_args)</code></p>
</li>
<li>
<p><code>TrainingAwareDataSparsity</code>: callback class to sparsify model during training. In addition to <code>data_sparsifier_class</code> and <code>data_sparsifier_args</code>,
    also accepts</p>
<ol>
<li><code>data_scheduler_class</code>: class/type of data scheduler to schedule the sparsity levels during training. Only the class should be passed, the object
will be created internally whenever necessary.</li>
<li><code>data_scheduler_args</code>: the arguments/config for the data scheduler constructor that will be used while creating the object.</li>
</ol>
<p>Example:</p>
<p>```
from data_sparsity import TrainingAwareDataSparsity
sparsifier_args = {
    'sparsity_level': 0.5,
    'sparse_block_shape': (1, 4),
    'zeros_per_block': 4
}
scheduler_args = {
    'gamma': 2,
    'step_size': 1
}</p>
<p>ta_callback = TrainingAwareDataSparsity(
    data_sparsifier_class=DataNormSparsifier,
    data_sparsifier_args=sparsifier_args,
    data_scheduler_class=StepSLScheduler,
    data_scheduler_args=scheduler_args
)
```</p>
</li>
</ol>
<p><strong>Note:</strong>
1. The model is copied and then sparsified, so the existing model is not modified.
2. The sparsified model can be accessed using <code>sparsified</code> attribute and can be used for comparison with the original version.
3. The data sparsifier/scheduler object will be created internally and will be attached to the model by the callback whenever necessary.</p>
<h2>Usage</h2>
<p>```
pl_module = SomePLModule()  # pl_module.model should specify the pytorch model</p>
<p>ds_callback = SomeDataSparsifierCallback(data_sparsifier_class=..., data_sparsifier_args=..., ...)  # add scheduler if TrainingAwareDataSparsifier
trainer = Trainer(callbacks=[ds_callback])</p>
<p>trainer.fit(pl_module, train_data_loader, val_data_loader)</p>
<h1>NOTE: pl_module.model is not sparsified</h1>
<h1>access sparsified model</h1>
<p>sparsified_model = ds_callback.sparsified
```</p>