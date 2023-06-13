<h1>Slurm Cluster Resolver</h1>
<p>The Slurm Cluster Resolver resolves cluster specification for distributing
TensorFlow work launched on HPC systems running on Slurm. This implementation is
able to handle homogeneous and heterogeneous tasks as long as the number of GPUs
per node and task are the same. This means on nodes with 4 GPUs each it will be
possible to allocate 4 processes on node A and only 2 on node B. The resolution
is done by determining job configuration through a number of Slurm variables and
can be overwritten by user input. By default everything is determined from the
Slurm environment, hence for most uses case no manual setting of parameters is
required.</p>
<h2>How it works</h2>
<p><code>SlurmClusterResolver</code> reads the environment variables that are set inside a job
step launched by Slurm. This means it will only work correctly for applications
launched via <code>srun</code>.</p>
<p>The process ID/rank is extracted from environment variable <code>SLURM_PROCID</code> and
the total number of tasks launched is extracted from <code>SLURM_STEP_NUM_TASKS</code>. The
hostnames are resolved by inspection <code>SLURM_STEP_NODELIST</code>. The number of tasks
per node is extracted from <code>SLURM_STEP_TASKS_PER_NODE</code>, unless a value is
specified by user. By using this variable heterogeneous task distributions are
possible. The user can set <code>tasks_per_node</code> to a single integer for homogeneous
tasks or a dictionary mapping node names to number of tasks for heterogeneous
distributions. However setting this is <strong>NOT</strong> recommended as there is a chance
it makes <code>SLURM_PROCID</code> be wrong.</p>
<p>A base port can be specified by user and in case there are more than one task
launched per node the port number will be incremented for each additional tasks
on that node. However a reasonable default is used.</p>
<p>The number of GPUs present on each node and number of GPUs for each tasks are
automatically detected. This is done by checking for <code>CUDA_VISIBLE_DEVICES</code>
first (which is set by Slurm to a list of GPUs for the current node) and has a
fallback to using <code>nvidia-smi</code>. If this doesn't work or non-NVIDIA GPUs are used
those 2 values have to be specified by the user. By default allocated GPUs will
be automatically exposed to processes according to specification by setting
<code>CUDA_VISIBLE_DEVICES</code>.</p>
<h2>Basic example</h2>
<ul>
<li>Slurm allocation in shell <code>salloc --nodes=2 -t 01:30:00 --ntasks-per-node=2
    --gres=gpu:k80:4 --exclusive</code></li>
<li>Run the example <code>srun python tf_example.py</code></li>
<li>Creating cluster in Python <code>import tensorflow as tf cluster_resolver =
    tf.distribute.cluster_resolver.SlurmClusterResolver() strategy =
    tf.distribute.experimental.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver)
    with strategy.scope(): # Load and compile model and data</code></li>
</ul>
<p>The above example will allocate 4 jobs on 2 nodes with each node having 2 jobs
and 4 GPUs. <code>cluster_resolver.cluster_spec()</code> will return a cluster
specification object in protobuf format with the following value (host names may
vary): <code>job { name: "worker" tasks { key: 0 value: "t02n13:8888" } tasks { key:
1 value: "t02n13:8889" } tasks { key: 2 value: "t02n41:8888" } tasks { key: 3
value: "t02n41:8889" } }</code></p>
<p>The <code>job_name</code> will be <code>worker</code> for all nodes and <code>task_index</code> will be <code>0</code> to
<code>3</code>. Also GPUs will be allocated automatically, so the first job on each node
will see GPU 0 and 1, and the second GPU 2 and 3.</p>
<h2>Advanced example</h2>
<ul>
<li>Assuming the same job parameters (<code>salloc</code> &amp; <code>srun</code>) as above</li>
<li>Creating cluster in Python ``` cluster_resolver =
    tf.contrib.cluster_resolver.SlurmClusterResolver( {'ps': 1, 'worker': 3},
    port_base=1337, tasks_per_node=2, gpus_per_node=2, gpus_per_task=1,
    auto_set_gpu=False)</li>
</ul>
<p>cluster = cluster_resolver.cluster_spec() job_name, task_index =
cluster_resolver.get_task_info() ```</p>
<p>In this case 1 parameter server job and 3 worker jobs are used. The resulting
protobuf specification will look similar to this: <code>job { name: "ps" tasks { key:
0 value: "t02n13:1337" } } job { name: "worker" tasks { key: 0 value:
"t02n13:1338" } tasks { key: 1 value: "t02n41:1337" } tasks { key: 2 value:
"t02n41:1338" } }</code></p>
<p>The value of <code>job_name</code> will be <code>ps</code> for <code>t02n13:1337</code> and <code>worker</code> for all
others. There will be no GPU allocation done by the cluster resolver, so this
has to be done manually which is useful if e.g. GPUs 0 should go to the first
process and GPU 3 to the second process on each node. Also note that only 1 GPU
will be used per task.</p>
<h2>Extension points</h2>
<p>The class <code>SlurmClusterResolver</code> provides some methods that are meant to be
overwritten by deriving classes:</p>
<ul>
<li><code>_resolve_own_rank</code></li>
<li><code>_resolve_num_tasks</code></li>
<li><code>_resolve_hostlist</code></li>
<li>
<p><code>_resolve_task_configuration</code></p>
<p>Those can be used to implement a cluster resolver that gets information from
a different source, e.g. via MPI, a file or other environment variables. See
the documentation of these methods on what to return.</p>
</li>
</ul>