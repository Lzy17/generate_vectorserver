<h1>Multi-Host HLO Runner</h1>
<p>[TOC]</p>
<p>This tool lets you run an HLO module on one or more GPUs.</p>
<h2>Running multi-GPU (sharded) HLOs</h2>
<p>We can identify these HLOs by seeing <code>sharding=</code> annotations. For example
<code>sharding={devices=[1,1,2,1]0,1}</code> means that the annotated tensor should be
sharded to 2 GPUs (GPU0 and GPU1) along the 3rd dimension.</p>
<p>The following instructions assume the working directory is the Tensorflow Git
repository and that it had been ./configure'd.</p>
<p>If we have enough GPUs, we can replay these HLOs like this:</p>
<p><code>bazel run -c opt --config=cuda --dynamic_mode=off \
  //tensorflow/compiler/xla/tools/multihost_hlo_runner:hlo_runner_main \
  -- --device_type=gpu --use_spmd_partitioning=true \
  --num_partitions=2 --num_replicas=1 \
  --hlo_file=my-hlo.txt</code></p>
<p>Tip: If the input generation takes too long or uses too much host memory,
consider using --hlo_argument_mode=uninitialized.</p>
<h3>Troubleshooting</h3>
<ul>
<li>Errors such as <code>Check failed: result.replicas &gt;= 1 (0 vs. 1)</code>:</li>
<li>We have to make sure that we have enough GPUs.</li>
<li><code>CUDA_VISIBLE_DEVICES</code> must be set correctly or not set at all.</li>
<li>Crashes:<ul>
<li>We may want to use <code>--dynamic_mode=off</code>.</li>
<li>CUDA and Cudnn should be set up correctly.</li>
</ul>
</li>
</ul>