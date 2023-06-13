<h1>PyTorch DTensor (Prototype Release)</h1>
<p>This folder contains the DTensor (a.k.a DistributedTensor) implementation in PyTorch.</p>
<h2>Introduction</h2>
<p>We propose distributed tensor primitives to allow easier distributed computation authoring in SPMD(Single Program Multiple Devices) paradigm. The primitives are simple but powerful when used to express tensor distributions with both sharding and replication parallelism strategies. This could empower native Tensor parallelism among other advanced parallelism explorations. For example, to shard a big tensor across devices with 3 lines of code:</p>
<p>```python
import torch
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor</p>
<h1>Create a mesh topology with the available devices:</h1>
<h1>1. We can directly create the mesh using elastic launcher,</h1>
<h1>2. If using mp.spawn, we need to initialize the world process_group first.</h1>
<h1>i.e. torch.distributed.init_process_group(backend="nccl", world_size=world_size)</h1>
<p>mesh = DeviceMesh("cuda", list(range(world_size)))
big_tensor = torch.randn(100000, 88)</p>
<h1>Shard this tensor over the mesh by sharding <code>big_tensor</code>'s 0th dimension over the 0th dimension of <code>mesh</code>.</h1>
<p>my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
```</p>
<h2>Motivation</h2>
<p>Today there are mainly three ways to scale up distributed training: Data Parallel, Tensor Parallel and Pipeline Parallel. Each of them works on a separate dimension where solutions have been built independently (i.e. PyTorch DDP, FSDP, ShardedTensor, PiPPy, etc.). When training really large models, users would like to use these technologies together (i.e. 3-D Parallelism), while the interoperability of the existing solutions are not great and often hard to use (i.e. users might want arbitrary combinations of the data parallel, tensor parallel and pipeline parallel). This is becoming an issue for users and one of the biggest reasons is that there is no common abstraction that build the bridge between different parallelism strategies.</p>
<p>An ideal scenario is that users could build their distributed program just like authoring in a single node/device, without worrying about how to do distributed training in a cluster, and our solutions could help them run distributed training in an efficient manner. For example, researchers just need to build the big transformer model, and PyTorch Distributed automatically figures out how to split the model and run pipeline parallel across different nodes, how to run data parallel and tensor parallel within each node. In order to achieve this, we need some common abstractions to distribute tensor values and distributed computations accordingly.</p>
<p>There're many recent works that working on tensor level parallelism to provide common abstractions, see the <code>Related Works</code> in the last section for more details. Inspired by <a href="https://arxiv.org/pdf/2105.04663.pdf">GSPMD</a>, <a href="https://arxiv.org/pdf/2110.15032.pdf">Oneflow</a> and <a href="https://www.tensorflow.org/guide/dtensor_overview">TF’s DTensor</a>, we introduce PyTorch DTensor as the next generation of ShardedTensor to provide basic abstractions for distributing storage and computation. It serves as one of the basic building blocks for distributed program translations and describes the layout of a distributed training program. With the DTensor abstraction, we can seamlessly build parallelism strategies such as tensor parallelism, DDP and FSDP.</p>
<h2>Value Proposition</h2>
<p>PyTorch DTensor primarily:
-   Offers a uniform way to save/load <code>state_dict</code> during checkpointing, even when there’re complex tensor storage distribution strategies such as combining tensor parallelism with parameter sharding in FSDP.
-   Enables Tensor Parallelism in eager mode. Compared to ShardedTensor, DistributedTensor allows additional flexibility to mix sharding and replication.
-   Serves as the entry point of an SPMD programming model and the foundational building block for compiler-based distributed training.</p>
<h2>PyTorch DTensor</h2>
<h3>DTensor API</h3>
<p>We offer both a lower level DistributedTensor API and a module level API to create a <code>nn.Module</code> with “distributed” parameters.</p>
<h4>Basic DTensor API Examples</h4>
<p>Here are some basic DTensor API examples that showcase:
1. How to construct a DTensor directly, to represent different types of sharding, replication, sharding + replication strategies.
2. How to create DTensor from a local <code>torch.Tensor</code>.
3. How to “reshard” an existing DTensor to a different DTensor with modified placement strategy or world size.</p>
<p>```python
import torch
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, Replicate, distribute_tensor, distribute_module</p>
<h1>construct a device mesh with available devices (multi-host or single host)</h1>
<p>device_mesh = DeviceMesh("cuda", [0, 1, 2, 3])</p>
<h1>if we want to do row-wise sharding</h1>
<p>rowwise_placement=[Shard(0)]</p>
<h1>if we want to do col-wise sharding</h1>
<p>colwise_placement=[Shard(1)]</p>
<p>big_tensor = torch.randn(888, 12)</p>
<h1>distributed tensor returned will be sharded across the dimension specified in placements</h1>
<p>rowwise_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=rowwise_placement)</p>
<h1>if we want to do replication across a certain device list</h1>
<p>replica_placement = [Replicate()]</p>
<h1>distributed tensor will be replicated to all four GPUs.</h1>
<p>replica_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=replica_placement)</p>
<h1>if we want to distributed a tensor with both replication and sharding</h1>
<p>device_mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])</p>
<h1>replicate across the first dimension of device mesh, then sharding on the second dimension of device mesh</h1>
<p>spec=[Replicate(), Shard(0)]
partial_replica = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=spec)</p>
<h1>create a DistributedTensor that shards on dim 0, from a local torch.Tensor</h1>
<p>local_tensor = torch.randn((8, 8), requires_grad=True)
rowwise_tensor = DTensor.from_local(local_tensor, device_mesh, rowwise_placement)</p>
<h1>reshard the current row-wise tensor to a colwise tensor or replicate tensor</h1>
<p>colwise_tensor = rowwise_tensor.redistribute(device_mesh, colwise_placement)
replica_tensor = colwise_tensor.redistribute(device_mesh, replica_placement)</p>
<p>```</p>
<h4>High level User Facing APIs</h4>
<p>Users can use DTensor tensor constructors directly to create a distributed tensor (i.e. <code>distributed.ones/empty</code>), but for existing modules like <code>nn.Linear</code> that are already having <code>torch.Tensor</code> as parameters, how to make them distributed parameters? We offer a way to directly distribute a <code>torch.Tensor</code> and a module level APIs to directly distribute the module parameters. Below is the high level API we introduce:</p>
<p><code>``python
def distribute_tensor(tensor: torch.Tensor, device_mesh: DeviceMesh=None, placements: List[Placement]=None):
    '''
    distribute the tensor according to device_mesh and placements,</code>tensor` could be a "meta" tensor.
    '''</p>
<p>def distribute_module(
    module: nn.Module,
    device_mesh: DeviceMesh=None,
    partition_fn: Callable[[str, nn.Module, DeviceMesh], ...]=None,
    input_fn: Callable[...., None]=None,
    output_fn: Callable[...., None]=None,
):
    '''
    This function converts all module parameters to distributed tensor parameters according to the <code>partition_fn</code> specified.
    It could also control the input/output of the module by specifying the <code>input_fn</code> and <code>output_fn</code>.
    '''
```</p>
<h4>High level API examples:</h4>
<p>```python
class MyModule(nn.Module):
    def <strong>init</strong>(self):
        super().<strong>init</strong>()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.relu = nn.ReLU()</p>
<pre><code>def forward(self, input):
    return self.relu(self.fc1(input) + self.fc2(input))
</code></pre>
<p>mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1], [2, 3]])</p>
<p>def shard_params(mod_name, mod, mesh):
    rowwise_placement = [Shard(0)]
    def to_dist_tensor(t): return distribute_tensor(t, mesh, rowwise_placement)
    mod._apply(to_dist_tensor)</p>
<p>sharded_module = distribute_module(MyModule(), mesh, partition_fn=shard_params)</p>
<p>def shard_fc(mod_name, mod, mesh):
    rowwise_placement = [Shard(0)]
    if mod_name == "fc1":
        mod.weight = torch.nn.Parameter(distribute_tensor(mod.weight, mesh, rowwise_placement))</p>
<p>sharded_module = distribute_module(MyModule(), mesh, partition_fn=shard_fc)</p>
<p>```</p>
<h2>Compiler and PyTorch DTensor</h2>
<p>DTensor provides efficient solutions for cases like Tensor Parallelism. But when using the DTensor's replication in a data parallel fashion, it might become observably slower compared to our existing solutions like DDP/FSDP. This is mainly because mainly because DDP/FSDP have a global view of the entire model architecture, thus could optimize for data parallel specifically, i.e. collective fusion and computation overlap, etc. In contract, DistributedTensor as a Tensor-like object can only optimize within individual tensor operations.</p>
<p>To improve efficiency of DTensor-based data parallel training, we are exploring a compiler-based solution on top of DTensor, which can extract graph information from user programs to expose more performance optimization opportunities.</p>
<h2>Related Works</h2>
<p>This work is mainly inspired by <a href="https://arxiv.org/pdf/2105.04663.pdf">GSPMD</a>, <a href="https://arxiv.org/pdf/2110.15032.pdf">Oneflow</a> and <a href="https://www.tensorflow.org/guide/dtensor_overview">TF’s DTensor</a>. All of these three works use a single “distributed tensor” concept for both replication and sharding, and the solutions could enable users to build up their distributed training program in a uniform SPMD programming model. Specifically:</p>
<p>GSPMD:
-   GSPMD is now the fundamental component of JAX/TensorFlow distributed training and enables various optimizations with the XLA compiler to allow users to train their models efficiently in a large scale setting.
-   Fundamentally, GSPMD have three types of sharding strategies within a tensor: “tiled”, “replicated”, “partially tiled” to represent sharding and replication.
-   At the core of GSPMD Partitioner, it utilizes the XLA compiler to do advanced optimizations, i.e. sharding propagation and compiler based fusion.
-   XLA mark_sharding API: PyTorch XLA’s <a href="https://github.com/pytorch/xla/pull/3476">mark_sharding</a> API uses <a href="https://github.com/pytorch/xla/issues/3871">XLAShardedTensor</a> abstraction (i.e. sharding specs) in PyTorch/XLA. Under the hood XLAShardedTensor is utilizing the GSPMD partitioner to enable SPMD style training on TPU.</p>
<p>OneFlow GlobalTensor:</p>
<ul>
<li>OneFlow is building up their own solution of the “GlobalTensor” concept, which is a variant form of GSPMD sharding, allowing users to explore different parallel strategies with GlobalTensor.</li>
<li>OneFlow also has three types of tensor, but they are slightly different from GSPMD: “split”, “broadcast”, and “partial sum”. They don’t use partially tiled and instead have a concept of partial sum to partition the values.</li>
</ul>
<p>TensorFlow DTensor:
-   <a href="https://www.tensorflow.org/guide/dtensor_overview">DTensor Concepts</a> is an extension of TensorFlow synchronous distributed training. its sharding API, supported features and its compilation passes with MLIR.
-   DTensor also allows sharding and replication on an n-d mesh like device network.
-   DTensor implements MLIR passes to do propagation and operator implementations.</p>
<p>There are also several cutting edge research fields that embeds tensor sharding as part of the system, i.e. <a href="https://arxiv.org/pdf/1909.08053.pdf">Megatron-LM</a> for tensor parallelism on Transformer based models. <a href="https://github.com/microsoft/DeepSpeed">DeepSpeed</a> for training large scale models with different optimization techniques on top of tensor sharding.</p>
<h3>Additional context</h3>
<p>RFC: https://github.com/pytorch/pytorch/issues/88838</p>
<p>We are gathering early feedbacks about this proposal. We have also posted this <a href="https://dev-discuss.pytorch.org/t/rfc-pytorch-distributedtensor/740">RFC</a> to the dev-discuss forum, please feel free to comment directly in the above issue or in the forum post. To see a complete design doc with additional details about DTensor, please refer to this <a href="https://docs.google.com/document/d/1nFeJ8NSFNhNlCkNgWK31ZGRqm1L9rd0i_XN_RprphaI/edit#heading=h.6sovjqv9jiqn">doc</a></p>