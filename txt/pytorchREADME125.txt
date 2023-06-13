<h1>Distributed RPC Reinforcement Learning Benchmark</h1>
<p>This tool is used to measure <code>torch.distributed.rpc</code> throughput and latency for reinforcement learning.</p>
<p>The benchmark spawns one <em>agent</em> process and a configurable number of <em>observer</em> processes. As this benchmark focuses on RPC throughput and latency, the agent uses a dummy policy and observers all use randomly generated states and rewards. In each iteration, observers pass their state to the agent through <code>torch.distributed.rpc</code> and wait for the agent to respond with an action. If <code>batch=False</code>, then the agent will process and respond to a single observer request at a time. Otherwise, the agent will accumulate requests from multiple observers and run them through the policy in one shot. There is also a separate <em>coordinator</em> process that manages the <em>agent</em> and <em>observers</em>.</p>
<p>In addition to printing measurements, this benchmark produces a JSON file.  Users may choose a single argument to provide multiple comma-separated entries for (ie: <code>world_size="10,50,100"</code>) in which case the JSON file produced can be passed to the plotting repo to visually see how results differ.  In this case, each entry for the variable argument will be placed on the x axis.</p>
<p>The benchmark results comprise of 4 key metrics:
1. <em>Agent Latency</em> - How long does it take from the time the first action request in a batch is received from an observer to the time an action is selected by the agent for each request in that batch.  If <code>batch=False</code> you can think of it as <code>batch_size=1</code>.
2. <em>Agent Throughput</em> - The number of request processed per second for a given batch.  Agent throughput is literally computed as <code>(batch_size / agent_latency)</code>.  If not using batch, you can think of it as <code>batch_size=1</code>.
3. <em>Observer Latency</em> - Time it takes from the moment an action is requested by a single observer to the time the response is received from the agent.  Therefore if <code>batch=False</code>, observer latency is the agent latency plus the transit time it takes for the request to get to the agent from the observer plus the transit time it takes for the response to get to the observer from the agent.  When <code>batch=True</code> there will be more variation due to some observer requests being queued in a batch for longer than others depending on what order those requests came into the batch in.
4. <em>Observer Throughput</em> - Number of requests processed per second for a single observer.  Observer Throughput is literally computed as <code>(1 / observer_latency)</code>.</p>
<h2>Requirements</h2>
<p>This benchmark depends on PyTorch.</p>
<h2>How to run</h2>
<p>For any environments you are interested in, pass the corresponding arguments to <code>python launcher.py</code>.</p>
<p><code>python launcher.py --world-size="10,20" --master-addr="127.0.0.1" --master-port="29501 --batch="True" --state-size="10-20-10" --nlayers="5" --out-features="10" --output-file-path="benchmark_report.json"</code></p>
<p>Example Output:</p>
<h2>```</h2>
<h2>PyTorch distributed rpc benchmark reinforcement learning suite</h2>
<p>master_addr : 127.0.0.1
master_port : 29501
batch : True
state_size : 10-20-10
nlayers : 5
out_features : 10
output_file_path : benchmark_report.json
x_axis_name : world_size
world_size | agent latency (seconds)     agent throughput            observer latency (seconds)  observer throughput
            p50    p75    p90    p95    p50    p75    p90    p95    p50    p75    p90    p95    p50    p75    p90    p95
10          0.002  0.002  0.002  0.002  4432   4706   4948   5128   0.002  0.003  0.003  0.003  407    422    434    443
20          0.004  0.005  0.005  0.005  4244   4620   4884   5014   0.005  0.005  0.006  0.006  191    207    215    220</p>