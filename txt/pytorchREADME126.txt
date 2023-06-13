<h1>RPC PS Benchmark</h1>
<h2>How to add your experiment</h2>
<ol>
<li>Data<ul>
<li>Create a data class and add it to the data directory</li>
<li>Update benchmark_class_helper.py to include your data class in the data_map</li>
<li>Add configurations to data_configurations.json in the configurations directory</li>
</ul>
</li>
<li>Model<ul>
<li>Create a model class and add it to the model directory</li>
<li>Update benchmark_class_helper.py to include your model class in the model_map</li>
<li>Add configurations to model_configurations.json in the configurations directory</li>
</ul>
</li>
<li>Trainer<ul>
<li>Create a trainer class and add it to the trainer directory</li>
<li>Update benchmark_class_helper.py to include your trainer class in the trainer_map</li>
<li>Add configurations to trainer_configurations.json in the configurations directory</li>
</ul>
</li>
<li>Parameter Server<ul>
<li>Create a parameter server class and add it to the parameter_servers directory</li>
<li>Update benchmark_class_helper.py to include your parameter_server class in the ps_map</li>
<li>Add configurations to parameter_server_configurations.json in the configurations directory</li>
</ul>
</li>
<li>Script<ul>
<li>Create a bash script for your experiment and add it to the experiment_scripts directory</li>
</ul>
</li>
<li>Testing<ul>
<li>Add a test method for your script to test_scripts.py</li>
</ul>
</li>
</ol>
<h2>Trainer class</h2>
<p>The trainer directory contains base classes to provide a starting point for implementing a trainer.
Inherit from a base class and implement your trainer. The benchmark has two requirements for trainers.</p>
<ol>
<li>
<p>It must implement a <strong>init</strong> method that takes rank, trainer_count, and ps_rref as arguments</p>
<p><code>python
def __init__(self, rank, trainer_count, ps_rref, backend, use_cuda_rpc):</code></p>
</li>
<li>
<p>It must implement a train method that takes model and data as arguments.</p>
<p><code>python
def train(self, model, data):</code></p>
</li>
</ol>
<h2>Parameter Server class</h2>
<p>The parameter_server directory contains base classes to provide a starting point for implementing a parameter server.
Inherit from a base class and implement your parameter server. The benchmark has two requirements for parameter servers.</p>
<ol>
<li>
<p>It must implement a <strong>init</strong> method that takes rank and ps_trainer_count as arguments</p>
<p><code>python
def __init__(self, rank, ps_trainer_count, backend, use_cuda_rpc):</code></p>
</li>
<li>
<p>It must implement a reset_state method</p>
<p><code>python
def reset_state(ps_rref):</code></p>
</li>
</ol>
<h2>Testing</h2>
<p>Use <code>pytest</code> to run the test methods added to test_scripts.py. To test all the scripts added use <code>pytest test_scripts.py</code>.</p>