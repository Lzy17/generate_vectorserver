<h1>FX Graph Mode Quantization Design Doc</h1>
<p>High Level FX Graph Mode Quantization Flow
```
float_model            QConfigMapping           BackendConfig
    \                          |                        /
     \                         |                      /
      \                        |                    /
(prepare_fx/prepare_qat_fx)                        /
—-------------------------------------------------------
|                         Fuse                         |
|                  QAT Module Swap                     |
|                 Insert Observers                     |
—-------------------------------------------------------
                              |
                      Calibrate/Train
                              |
(convert_fx)                  |
—--------------------------------------------------------
|                         Convert                       |
|                        Lowering                       |
—--------------------------------------------------------
                              |
                       Quantized Model</p>
<p>```
Please refer to [TODO: link] for definitions of terminologies.</p>
<h2>Overview</h2>
<p>The FX graph representation is pretty close to python/eager mode, it preserves many python/eager mode constructs like modules, functionals, torch ops, so overall the implementation reuses some of building blocks and utilities from eager mode quantization, this includes the QConfig, QConfig propagation (might be removed), fused modules, QAT module, quantized modules, QAT module swapping utility. Also the overall flow exactly matches eager mode quantization, the only difference is that the transformations like fusion, inserting stubs are fully automated and controlled by QConfigMapping and BackendConfig.</p>
<h2>High Level Flow with Simple Example</h2>
<p><code>prepare_fx</code>:
<code>Floating Point Model --&gt; (1.1 `_fuse_fx`) --&gt; Fused Model
                     --&gt; (1.2 QAT Module Swap) --&gt; Model with QAT modules
                     --&gt; (1.3 Insert Observers) --&gt; Prepared Model</code></p>
<p><code>convert_fx</code>:
<code>Prepared Model --&gt; (2.1 `convert_to_reference`) --&gt; Reference Quantized Model
               --&gt; (2.2 Lower to Native Backend) --&gt; Quantized Model</code></p>
<p>In the following, I’ll first have a detailed description for each step, and then talk about the corresponding settings in BackendConfig. We’ll follow the terminologies defined in (draft) README.md of quantization syntax transforms in this doc.</p>
<h3>0. Original Model</h3>
<p>```
class LinearReLUModule(torch.nn.Module):
   def <strong>init</strong>(self):
       super().<strong>init</strong>()
       self.linear = torch.nn.Linear(5, 10).float()
       self.relu = torch.nn.ReLU()</p>
<p>def forward(self, x):
       return self.relu(self.linear(x))
```</p>
<h3>1.1 Fusion</h3>
<p>```
fused: GraphModule(
  (linear): LinearReLU(
    (0): Linear(in_features=5, out_features=10, bias=True)
    (1): ReLU()
  )
)</p>
<p>def forward(self, x):
    linear = self.linear(x);  x = None
    return linear
```</p>
<p>What we did in this example are:</p>
<ul>
<li>Identify (Linear - ReLU) subgraph by searching through the model graph</li>
<li>For each of the identified subgraph, we replace the <code>root_node</code> (typically the weighted module in the pattern, like Linear), with a fused module by calling the fuser_method for this pattern, a fused module is a sequential of a few modules, e.g. nni.LinearReLU is a sequential of linear and relu module</li>
</ul>
<p><code>backend_config</code> configurations relevant to this step are:</p>
<p>```
def fuse_linear_relu(is_qat, linear, relu):
    return nni.LinearReLU(linear, relu)</p>
<p>BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))
    .set_fuser_method(fuse_linear_relu)
    ._set_root_node_getter(my_root_node_getter)
    ._set_extra_inputs_getter(my_extra_inputs_getter)
```</p>
<p><code>BackendPatternConfig</code> takes in a pattern that specifies the fusion pattern that we want to search for, pattern format can be found in https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md</p>
<p><code>set_dtype_configs</code>: dtype_configs are used to check against the qconfig for the pattern, to see if the qconfig is supported in the target backend or not. Currently it’s not used in fusion, but we can add this check in the future, or remove this and always fuse these patterns.
<code>set_fuser_method</code>: specifies the fuser method to use for the pattern, a fuser method will take the matched object and fuse them into a fused module.
<code>_set_root_node_getter</code>: sets a function that takes a node pattern and returns the root node in the pattern.
<code>_set_extra_inputs_getter</code>: all input args of root node will be copied over to fused module, if there are extra inputs, this function will return a list of extra inputs given the pattern.</p>
<p>Example usage of <code>root_node_getter</code> and <code>extra_input_getter</code>: https://gist.github.com/jerryzh168/8bea7180a8ba3c279f2c9b050f2a69a6</p>
<h3>1.2 QAT Module Swap</h3>
<p>```
GraphModule(
  (linear): LinearReLU(
    in_features=5, out_features=10, bias=True
    (weight_fake_quant): MinMaxObserver(min_val=inf, max_val=-inf)
  )
)</p>
<p>def forward(self, x):
    linear = self.linear(x);  x = None
    return linear
```</p>
<p>In this step we swap the fused module to qat module, for example, swap nn.intrinsic.LinearReLU instances to nn.intrinsic.qat.LinearReLU module where we fake quantize the weight of linear.
For modules that has corresponding QAT modules we’ll call eager mode <code>convert</code> function with a mapping from float module to QAT module which will swap all float module (and fused module) with QAT module, this step is exactly the same as eager mode quantization, just called inside the <code>prepare_fx/prepare_qat_fx</code> function.</p>
<p><code>backend_config</code> configurations relevant in this step are:
<code>BackendPatternConfig(nni.LinearReLU)
    .set_qat_module(nniqat.LinearReLU)</code></p>
<p>The pattern used to initialize BackendPatternConfig is the class type for original or fused floating point module class.
<code>set_qat_module</code> sets the qat module class corresponding to the module class specified in the pattern.</p>
<h3>1.3 QuantDeQuantStub and Observer/FakeQuantize Insertion</h3>
<p>```
GraphModule(
  (activation_post_process_0): MinMaxObserver(min_val=inf, max_val=-inf)
  (linear): LinearReLU(
    (0): Linear(in_features=5, out_features=10, bias=True)
    (1): ReLU()
  )
  (activation_post_process_1): MinMaxObserver(min_val=inf, max_val=-inf)
)</p>
<p>def forward(self, x):
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
    linear = self.linear(activation_post_process_0);  activation_post_process_0 = None
    activation_post_process_1 = self.activation_post_process_1(linear);  linear = None
    return activation_post_process_1
```</p>
<p>Note: activation_post_process_0 and activation_post_process_1 will be updated with QuantDeQuantStub</p>
<p>QuantDeQuantStubs are inserted based on the <code>qconfig_mapping</code> provided by users. Also we have a backend_config that specifies the configs that are supported by the backend. In this step, we will
* Check if <code>qconfig_mapping</code> is compatible with <code>backend_config</code> or not, if user requested a qconfig that is not compatible with <code>backend_config</code>, we’ll not insert observers for the operator, the config would just be ignored.
* Insert observer for the input and output of the subgraph, based on the <code>qconfig_mapping</code> (what user requested) and the <code>backend_config</code> (how the operator should be observed in a backend).</p>
<p>Detailed walkthrough for this step in <code>prepare_qat_fx</code> (inserting QDQStub and FakeQuantize modules):
Note: We could also insert QStub and DQStub in this step when users request to change the interface dtype for the model, standalone module or custom modules.
```</p>
<h1>fused and qat swapped model</h1>
<h1>graph 1:</h1>
<p>input - qat_linear_relu - output
              |
          FakeQuantize
(need to be updated with QDQStub + FakeQuantize)
              |
           weight</p>
<h1>qconfig_mapping (simplified, shown as dict)</h1>
<p>{'qat_linear_relu': QConfig(
  weight=MinMaxObserver.with_args(dtype=torch.qint8),
  activation=HistogramObserver.with_args(dtype=torch.quint8),
)}</p>
<h1>backend_config (simplified)</h1>
<p>{
  'pattern': nnqat.LinearReLU,
  'dtype_configs': [{input: torch.quint8, output: torch.quint8, weight: torch.qint8}],
}
```</p>
<p>step 1: assign qconfig to each op (please see [TODO: link] for details)</p>
<p>step 2: determine which qconfigs are valid according to the backend configuration (please see [TODO: link] for details)
(we should add a warning here)</p>
<p>step 3: for subgraphs with validated qconfigs, insert qstub/dqstub/qdqstub needed</p>
<p>To talk about what happens in this step, let’s first define some terms. Let’s view the computation graph we showed above as a Graph consists of nodes and edges, each node here will be an FX Node that represents some computation, for example linear, and each edge will be a connection between two nodes, and each edge can both be viewed as the output of the previous Node or the input of the next Node.</p>
<p>The end goal for this step is to insert QDQStubs at edges so that we produce a graph of quantized reference model when each QDQStub represents a quantize operator followed by a dequantize operator.</p>
<p>```</p>
<h1>graph 2:</h1>
<p>input - QDQStub1 (FakeQuantize) - qat_linear_relu - QDQStub2 (FakeQuantize) - output
                                      |
                                FakeQuantize
                  (need to be updated with QDQStub + FakeQuantize)
                                      |
                                    weight
```
Note: weight + FakeQuantize is a part of qat_linear_relu</p>
<p>The overall logic to insert QDQStub1 and QDQStub2 inplace is the following:
0. For each node in the original graph, we compute the target_dtype for input and output for it based on qconfig, for graph1, configured with qconfig_mapping, we have:
```</p>
<h1>node_name_to_target_dtype_info =</h1>
<h1>{</h1>
<h1># this is placeholder node in FX Graph</h1>
<h1>"input" : {"input_activation": torch.float32, "output_activation": torch.float32},</h1>
<h1>"qat_linear_relu": {"input_activation": torch.quint8, "output_activation": torch.quint8, "weight": ...}</h1>
<h1># this is the return node in FX Graph</h1>
<h1>"output": {"input_activation": torch.float32, "output_activation": torch.float32}</h1>
<h1>}</h1>
<p>```
Note: this map is generated before we insert qdqstub to graph1, and will not change in the process.</p>
<ol>
<li>Inserting QDQStub1 (for input of qat_linear_relu)
   We need to look at the edge between <code>input</code> Node and <code>qat_linear_relu</code> Node here, we need to decide if we need to insert a
   QDQStub at this edge, which could serve as an input argument for <code>qat_linear_relu</code> Node (and also output for <code>input</code> Node)
   The way we decide if we want to insert QDQStub here is to figure out</li>
</ol>
<p>(1). The target dtype for output of <code>input</code> Node, which is torch.float32</p>
<p>(2). The target dtype for input of <code>qat_linear_relu</code> Node, which is torch.quint8
   There is a mismatch here and (2) is a quantized dtype, so we need to insert QDQStub at the edge.</p>
<p>We also need to attach observer/fakequant module to the QDQStub we inserted here.
2. Insert QDQStub2 (for output of qat_linear_relu)
   The logic for inserting QDQStub for output is much easier, since we assume all modules/functions in the graph produce fp32 output
   by default (we can have additional checks and extend this to work for other dtypes after we have type inference ready),
   we just need to look at the target output dtype for qat_linear_relu Node, and if it is a quantized dtype (quint8, qint8, float16),
   we would insert a QDQStub here.</p>
<p>Questions: How to avoid inserting duplicate QDQStubs?
e.g. when we have a single input being used by multiple ops:
<code>input — linear1 —-
     \--- linear2 —</code>
how do we make sure we only insert one QDQStub for input of both linear1 and linear2?
<code>input - QDQStub — linear1 -
             \ —- linear2 -</code></p>
<p>The way we do it right now is before we insert QDQStub, we look at all users of <code>input</code> Node here and make sure there is no QDQStubs
with the same target_dtype, that is, if we already inserted a QDQStub with dtype quint8 for linear1, and linear2 is also connected to it, if we request another QDQStub with dtype quint8 when processing linear2 Node, we’ll detect that the desired QDQStub already exists and do nothing</p>
<p>Question: What is the logic for keeping output to be float32?
Let’s say the output of <code>qat_linear_relu</code> Node is configured as float32, both in qconfig_mapping and backend_config:
```</p>
<h1>qconfig_mapping (simplified, shown as dict)</h1>
<p>{'qat_linear_relu': QConfig(
  weight=MinMaxObserver.with_args(dtype=torch.qint8),
  input_activation=HistogramObserver.with_args(dtype=torch.quint8),
  output_activation=PlaceholderObserver.with_args(dtype=torch.float32),
)}</p>
<h1>backend_config (simplified)</h1>
<p>{
  'pattern': nnqat.LinearReLU,
  'dtype_configs': [{input: torch.quint8, output: torch.float32, weight: torch.qint8}],
}
```</p>
<p>What we’ll do here is when we are trying to insert output QDQStub for <code>qat_linear_relu</code>, we look at the target output dtype for this node (node_name_to_target_dtype_info["qat_linear_relu"]["output_activation"], and find that it is float, which is not a quantized dtype, so
will do nothing here.
Note that this does not prevent other operators following <code>qat_linear_relu</code> to insert a QDQStub at the output of <code>qat_linear_relu</code>, since we are dealing with an <code>edge</code> of the graph here, and an <code>edge</code> is connected to two nodes, which means
the output of <code>qat_linear_relu</code> will also be the input of a node following <code>qat_linear_relu</code>.</p>
<p><code>backend_config</code> configurations used in this step:
<code>BackendConfig(nniqat.LinearReLU)
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
    .set_dtype_configs([
        DTypeConfig(input_dtype=torch.quint8, output_dtype = torch.quint8, weight_dtype = torch.qint8, bias_dtype = torch.float32)]
    )</code></p>
<p>Pattern in this case is the same as before, it defines the pattern for the subgraph we are dealing with</p>
<p><code>set_observation_type</code>: sets the observation type for the patter, currently only two types:</p>
<p><code>OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT</code> means the output observer instance will be different from the input, which is the most common type of observer placement.</p>
<p><code>OUTPUT_SHARE_OBSERVER_WITH_INPUT</code> means the output observer is shared with input, they will be the same instance. This is useful for operators like cat.</p>
<p><code>set_dtype_configs</code>: sets a list of supported (activation, weight, bias, etc.) dtype combinations for qconfigs for the pattern. Note that we represent different modes of quantization (static/dynamic/<code>weight_only</code>) purely through this combination, for example, fbgemm static quantization can be represented as:
<code>{
  "input_activation": torch.quint8,
  "weight": torch.qint8,
  "output_activation": torch.quint8
}</code></p>
<p>Note: the dtype config will be used to configure the support for dynamic quantization as well</p>
<p>Note: we may extend this to support more fine grained configurations of args, kwargs, attributes and outputs in the future</p>
<p>Note: we are referring to observer here, which is an implementation detail, we can change this to talk about quantization parameters instead, e.g. <code>QParamsType.OUTPUT_USE_DIFFERENT_QPARAMS_AS_INPUT</code> and <code>QParamsType.OUTPUT_USE_SAME_QPARAMS_AS_INPUT</code></p>
<h3>2. Calibration/Training</h3>
<p>After we insert observers, we run the model to calibrate observers or to fine tune. This step is identical to eager mode quantization. After that the observer/fakequantize modules contain sufficient information to determine quantization parameters according to the observed data.</p>
<h3>3.1 Conversion to Reference Quantized Model</h3>
<p>```
quantized: GraphModule(
  (linear): LinearReLU(
    (0): QuantizedLinear(Reference)(in_features=5, out_features=10, bias=True)
    (1): ReLU()
  )
)</p>
<p>def forward(self, x):
    linear_input_scale_0 = self.linear_input_scale_0
    linear_input_zero_point_0 = self.linear_input_zero_point_0
    quantize_per_tensor = torch.quantize_per_tensor(x, linear_input_scale_0, linear_input_zero_point_0, torch.quint8);  x = linear_input_scale_0 = linear_input_zero_point_0 = None
    dequantize = quantize_per_tensor.dequantize();  quantize_per_tensor = None
    linear = self.linear(dequantize);  dequantize = None
    linear_scale_0 = self.linear_scale_0
    linear_zero_point_0 = self.linear_zero_point_0
    quantize_per_tensor_1 = torch.quantize_per_tensor(linear, linear_scale_0, linear_zero_point_0, torch.quint8);  linear = linear_scale_0 = linear_zero_point_0 = None
    dequantize_1 = quantize_per_tensor_1.dequantize();  quantize_per_tensor_1 = None
    return dequantize_1
```</p>
<p>After we insert observers, we’ll need to convert the model to a reference quantized model. Reference quantized model is a model that uses reference patterns to represent quantized operators, this serves as the standard interface for quantized operators between PyTorch quantization and backend lowering passes. For more details, please take a look at this <a href="https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md">RFC</a>. This pass is pretty straightforward, what we do is:</p>
<p>(1). for each QDQStub (attached with Observer for FakeQuantize modules) in the graph, we'll convert it to calls to quantize and dequantize functions based on the attributes of attached Observer and FakeQuantize modules (e.g. qscheme, dtype etc.)</p>
<p>(2). for weighted modules like linear/conv, we convert them to corresponding reference quantized module.</p>
<p>Example:
```</p>
<h1>graph 1</h1>
<p>input - QDQStub1 (FakeQuantize) - qat_linear_relu - QDQStub2 (FakeQuantize) - output
                                      |
                                FakeQuantize
                  (need to be updated with QDQStub + FakeQuantize)
                                      |
                                    Weight</p>
<p>Note: weight + FakeQuantize is a part of qat_linear_relu module</p>
<h1>graph 2</h1>
<p>input - quantize - dequantize - reference_linear_relu - quantize - dequantize - output
                                        |
                                   dequantize
                                        |
                                    quantize
                                        |
                                      weight
```
Note: weight + quantize + dequantize is a part of reference_linear_relu module</p>
<p>To decide which quantize node we want to use, we’ll look at:</p>
<p>(1). dtype of attached Observer/FakeQuantize module</p>
<p>(2). qscheme of attached Observer/FakeQuantize module</p>
<p>(3). (optionally) other attributes of attached Observer/FakeQuantize module</p>
<p>The quantize operator we can choose from right now are: (quantize_per_tensor, quantize_per_channel, to, quantize_per_tensor_dynamic)</p>
<p><code>backend_config configurations used in this step:
BackendConfig(nniqat.LinearReLU)
    .set_root_module(nn.Linear)
    .set_reference_quantized_module_for_root(nnqr.Linear)
    .set_fused_module(nni.LinearReLU)</code></p>
<p>Pattern in this case is the same as before, it defines the pattern for the subgraph we are dealing with</p>
<p><code>set_root_module</code>: Sets a module class for the root of the pattern, e.g. nn.Linear for a nni.LinearReLU/nniqat.LinearReLU, used to identify the modules that needs to be swapped to reference quantized module</p>
<p><code>set_reference_quantized_module_for_root</code>: Sets the corresponding reference quantized module class for root module class, e.g. when root_module is nn.Linear, this will be nn.quantized.reference.Linear, used to swap the root module to be a reference quantized module.</p>
<p>Note: we are only swapping <code>root_module</code> here, for example, in the current example, the original module is <code>nniqat.LinearReLU</code>, when we are converting weight modules(step (2)), we first convert <code>nniqat.LinearReLU</code> to a float module, in this case, the fused LinearReLU module: <code>nni.LinearReLU</code>, and then swap the root_module (<code>nn.Linear</code>) with reference quantized module (<code>nnqr.Linear</code>), so we end up with a <code>nni.LinearReLU</code> module, which is a sequential module of a <code>nnqr.Linear</code> and <code>nn.ReLU</code>.</p>
<p>Basically, the corresponding reference quantized module for both <code>nniqat.LinearReLU</code> and <code>nni.LinearReLU</code> would be a <code>nni.LinearReLU</code> Sequential module (originally <code>nn.Linear</code> + <code>nn.ReLU</code>) with <code>nn.Linear</code> being replaced by <code>nnqr.Linear</code>: <code>nni.LinearReLU(nnqr.Linear, nn.ReLU)</code>.</p>
<p><code>set_fused_module</code>: This is the corresponding fused module class for the pattern, used to identify fused modules that needs to be converted to reference quantized module</p>
<h3>3.2 Lower to PyTorch Native Backend</h3>
<p>```
GraphModule(
  (linear): QuantizedLinearReLU(in_features=5, out_features=10, scale=1.0, zero_point=0, qscheme=torch.per_tensor_affine)
)</p>
<p>def forward(self, x):
    linear_input_scale_0 = self.linear_input_scale_0
    linear_input_zero_point_0 = self.linear_input_zero_point_0
    quantize_per_tensor = torch.quantize_per_tensor(x, linear_input_scale_0, linear_input_zero_point_0, torch.quint8);  x = linear_input_scale_0 = linear_input_zero_point_0 = None
    linear = self.linear(quantize_per_tensor);  quantize_per_tensor = None
    dequantize_1 = linear.dequantize();  linear = None
    return dequantize_1
```</p>
<p>Currently, PyTorch has native quantized backends: fbgemm and qnnpack, so we need a lowering pass to lower the reference quantized model to a model that is using native quantized operators in PyTorch. What this pass did is</p>
<ol>
<li>
<p>Recognize the reference patterns like: "dequantize - <code>float_op</code> - quantize" in the graph and replace them with the quantized modules (under torch.nn.quantized namespace) or operators (under torch.ops.quantized namespace, or torch namespace)
In general there are three types of patterns:</p>
</li>
<li>
<p>Static quantization:
<code>dequantize -&gt; float_op -&gt; quantize_per_tensor</code></p>
</li>
<li>
<p>Dynamic quantization:
<code>quantize_per_tensor_dynamic -&gt; dequantize -&gt; float_op</code></p>
</li>
<li>
<p>Weight only quantization:
<code>input - float_op - output
      weight - quantize_per_tensor - dequantize /</code></p>
</li>
<li>
<p>Prepack and fold the weights for quantized linear and quantized conv operator</p>
</li>
<li>The lowering pass is also going to keep some patterns for quantized operators unfused, since user may explicitly request some operators to stay in float by configuring the qconfig to be None</li>
</ol>
<p>There are no configurations related to lowering in <code>backend_config</code> since it is backend developer’s responsibility to implement lowering pass and each of the backend developers may have their own configurations. So from end to end, <code>backend_config</code> and together with qconfig_mapping controls what Reference Quantized Model is produced by FX Graph Mode Quantization, not lowered model.</p>
<p>However, for some operator based backends, like the current pytorch native backends including fbgemm and qnnpack. We could interpret <code>backend_config</code> in terms of configurations for operators as well. e.g. configuring <code>input_dtype=quint8</code>, <code>weight_dtype=qint8</code>, <code>output_dtype=torch.quint8</code> for nn.Linear is saying that the quantized linear will take a <code>quint8</code> activation and <code>qint8</code> weight as input and outputs a <code>quint8</code> activation. But there is no guarantee that this interpretation will always work in the future, especially when we add new flavors of quantized operators.</p>
<h2>Extensibility</h2>
<p>FX graph mode quantization can be extended to work with different backends, which may have different sets of supported quantized operator patterns and different requirements for each pattern. For more detail, please refer to the <a href="/torch/ao/quantization/backend_config/README.md">BackendConfig README</a>.</p>