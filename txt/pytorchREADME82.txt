<h2>BackendConfig Overview</h2>
<p>BackendConfig allows PyTorch quantization to work with different backend or kernel libraries. These backends may have different sets of supported quantized operator patterns, and the same operator patterns may require different handling across different backends. To make quantization work with different backends and allow maximum flexibility, we strived to make all the parts of the quantization flow configurable with BackendConfig. Currently, it is only used by FX graph mode quantization. For more details on how it integrates with the FX graph mode quantization flow, refer to this <a href="/torch/ao/quantization/fx/README.md">README</a>.</p>
<p>BackendConfig configures quantization behavior in terms of operator patterns. For each operator pattern, we need to specify what the supported data types are for the input and output activations, weights, and biases, and also specify the QAT modules, the reference quantized modules etc., which will be used in module swapping during the quantization passes.</p>
<p>Quantized backends can have different support in terms of the following aspects:
* Quantization scheme (symmetric vs asymmetric, per-channel vs per-tensor)
* Data type (float32, float16, int8, uint8, bfloat16, etc.) for input/output/weight/bias
* Quantized (and fused) mapping: Some quantized operators may have different numerics compared to a naive (dequant - float_op - quant) reference implementation. For weighted operators, such as conv and linear, we need to be able to specify custom reference modules and a mapping from the float modules
* QAT mapping: For weighted operators, we need to swap them with the Quantization Aware Training (QAT) versions that add fake quantization to the weights</p>
<p>As an example, here is what fbgemm looks like:
|                                           | fbgemm                                                                |
|-------------------------------------------|-----------------------------------------------------------------------|
| Quantization Scheme                       | activation: per tensor, weight: per tensor or per channel             |
| Data Type                                 | activation: quint8 (with qmin/qmax range restrictions), weight: qint8 |
| Quantized and Fused Operators and Mapping | e.g. torch.nn.Conv2d -&gt; torch.ao.nn.quantized.reference.Conv2d        |
| QAT Module Mapping                        | e.g. torch.nn.Conv2d -&gt; torch.ao.nn.qat.Conv2d                        |</p>
<p>Instead of hardcoding the fusion mappings, float to reference quantized module mappings, fusion patterns etc., we will derive everything from the BackendConfig throughout the code base. This allows PyTorch Quantization to work with all first-party (fbgemm and qnnpack) and third-party backends (TensorRT, executorch etc.) that may differ from native backends in different aspects. With the recent addition of xnnpack, integrated as part of the qnnpack backend in PyTorch, the BackendConfig is needed to define the new constraints required for xnnpack quantized operators.</p>
<h2>Pattern Specification</h2>
<p>The operator patterns used in BackendConfig are float modules, functional operators, pytorch operators, or a tuple combination of the above. For example:
* torch.nn.Linear
* torch.nn.functional.linear
* torch.add
* operator.add
* (torch.nn.functional.linear, torch.nn.functional.relu)
* (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU)</p>
<p>Tuple patterns are treated as sequential patterns, and currently only tuples of 2 or 3 elements are supported.</p>
<h3>Advanced Pattern Specification</h3>
<p>The above format should satisfy the vast majority of use cases. However, it does not handle more complex scenarios such as graph patterns. For these use cases, the BackendConfig API offers an alternative "reverse nested tuple" pattern format, enabled through <code>BackendPatternConfig()._set_pattern_complex_format(...)</code>. Note that this format is deprecated and will be replaced in a future version of PyTorch.
<code>operator = module_type | functional | torch op | native op | MatchAllNode
Pattern = (operator, Pattern, Pattern, ...) | operator</code>
where the first item for each Pattern is the operator, and the rest are the patterns for the arguments of the operator.
For example, the pattern (nn.ReLU, (operator.add, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d))) would match the following graph:
<code>tensor_1            tensor_2
 |                    |
 *(MatchAllNode)  nn.Conv2d
 |                    |
 |             nn.BatchNorm2d
 \                  /
  -- operator.add --
         |
      nn.ReLU</code></p>
<p>During prepare and convert, we’ll match the last node, which will be the anchor point of the match, and we can retrieve the whole graph by tracing back from the node. E.g. in the example above, we matched the <code>nn.ReLU</code> node, and <code>node.args[0]</code> is the <code>operator.add</code> node.</p>
<h2>BackendConfig Implementation</h2>
<p>The BackendConfig is comprised of a list of BackendPatternConfigs, each of which define the specifications and the requirements for an operator pattern. Here is an example usage:</p>
<p>```
import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)</p>
<p>weighted_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float)</p>
<p>def fuse_conv2d_relu(is_qat, conv, relu):
    """Return a fused ConvReLU2d from individual conv and relu modules."""
    return torch.ao.nn.intrinsic.ConvReLU2d(conv, relu)</p>
<h1>For quantizing Linear</h1>
<p>linear_config = BackendPatternConfig(torch.nn.Linear) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_root_module(torch.nn.Linear) \
    .set_qat_module(torch.ao.nn.qat.Linear) \
    .set_reference_quantized_module(torch.ao.nn.quantized.reference.Linear)</p>
<h1>For fusing Conv2d + ReLU into ConvReLU2d</h1>
<p>conv_relu_config = BackendPatternConfig((torch.nn.Conv2d, torch.nn.ReLU)) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_fused_module(torch.ao.nn.intrinsic.ConvReLU2d) \
    .set_fuser_method(fuse_conv2d_relu)</p>
<h1>For quantizing ConvReLU2d</h1>
<p>fused_conv_relu_config = BackendPatternConfig(torch.ao.nn.intrinsic.ConvReLU2d) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_root_module(torch.nn.Conv2d) \
    .set_qat_module(torch.ao.nn.intrinsic.qat.ConvReLU2d) \
    .set_reference_quantized_module(torch.ao.nn.quantized.reference.Conv2d)</p>
<p>backend_config = BackendConfig("my_backend") \
    .set_backend_pattern_config(linear_config) \
    .set_backend_pattern_config(conv_relu_config) \
    .set_backend_pattern_config(fused_conv_relu_config)
```</p>
<h3>Observer Insertion</h3>
<p>Relevant APIs:
* <code>set_observation_type</code></p>
<p>During the prepare phase, we insert observers (or QuantDeQuantStubs in the future) into the graph for this operator pattern based on the observation type, which specifies whether to use different observers for the inputs and the outputs of the pattern. For more detail, see <code>torch.ao.quantization.backend_config.ObservationType</code>.</p>
<h3>Reference Quantized Patterns</h3>
<p>Relevant APIs:
* <code>set_root_module</code>
* <code>set_reference_quantized_module</code></p>
<p>During the convert phase, when we construct the reference quantized model, the root modules (e.g. <code>torch.nn.Linear</code> for <code>nni.LinearReLU</code> or <code>nniqat.LinearReLU</code>) will be swapped to the corresponding reference quantized modules (e.g. <code>torch.ao.nn.reference.Linear</code>). This allows custom backends to specify custom reference quantized module implementations to match the numerics of their lowered operators. Since this is a one-to-one mapping, both the root module and the reference quantized module must be specified in the same BackendPatternConfig in order for the conversion to take place.</p>
<h3>Fusion</h3>
<p>Relevant APIs:
* <code>set_fuser_method</code>
* <code>set_fused_module</code>
* <code>_set_root_node_getter</code>
* <code>_set_extra_inputs_getter</code></p>
<p>As an optimization, operator patterns such as (<code>torch.nn.Linear</code>, <code>torch.nn.ReLU</code>) may be fused into <code>nni.LinearReLU</code>. This is performed during the prepare phase according to the function specified in <code>set_fuser_method</code>, which replaces the pattern with the fused module. During the convert phase, these fused modules (identified by <code>set_fused_module</code>) will then be converted to the reference quantized versions of the modules.</p>
<p>In FX graph mode quantization, we replace the corresponding nodes in the graph using two helper functions set by the user: <code>root_node_getter</code>, which returns the root node (typically the weighted module in the pattern like <code>torch.nn.Linear</code>) to replace the matched pattern in the graph, and <code>extra_inputs_getter</code>, which returns a list of extra input arguments that will be appended to the existing arguments of the fused module (copied over from the root node). See <a href="https://gist.github.com/jerryzh168/8bea7180a8ba3c279f2c9b050f2a69a6">this snippet</a> for an example usage.</p>
<h3>Data Type Restrictions</h3>
<p>Relevant APIs:
* <code>add_dtype_config</code>
* <code>set_dtype_configs</code></p>
<p>DTypeConfig specifies a set of supported data types for input/output/weight/bias along with the associated constraints, if any. There are two ways of specifying <code>input_dtype</code>, <code>output_dtype</code>, and <code>weight_dtype</code>, as simple <code>torch.dtype</code>s or as <code>DTypeWithConstraints</code>, e.g.:</p>
<p>```
import torch
from torch.ao.quantization.backend import DTypeConfig, DTypeWithConstraints</p>
<p>dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float)</p>
<p>dtype_config_with_constraints = DTypeConfig(
    input_dtype=DTypeWithConstraints(
        dtype=torch.quint8,
        quant_min_lower_bound=0,
        quant_max_upper_bound=255,
        scale_min_lower_bound=2 <strong> -12,
    ),
    output_dtype=DTypeWithConstraints(
        dtype=torch.quint8,
        quant_min_lower_bound=0,
        quant_max_upper_bound=255,
        scale_min_lower_bound=2 </strong> -12,
    ),
    weight_dtype=DTypeWithConstraints(
        dtype=torch.qint8,
        quant_min_lower_bound=-128,
        quant_max_upper_bound=127,
        scale_min_lower_bound=2 ** -12,
    ),
    bias_dtype=torch.float)
```</p>
<p>During the prepare phase of quantization, we will compare the data types specified in these DTypeConfigs to the ones specified in the matching QConfig for a given operator pattern. If the data types do not match (or the constraints are not satisfied) for all the DTypeConfigs specified for the operator pattern, then we will simply ignore the QConfig and skip quantizing this pattern.</p>
<h4>Quantization range</h4>
<p>The user's QConfig may specify <code>quant_min</code> and <code>quant_max</code>, which are min and max restrictions on the quantization values. Here we set the lower bound for the <code>quant_min</code> and then upper bound for the <code>quant_max</code> to represent the limits of the backend. If a QConfig exceeds these limits in either direction, it will be treated as violating this constraint.</p>
<h4>Scale range</h4>
<p>Similarly, the user's QConfig may specify a minimum value for the quantization scale (currently exposed as <code>eps</code> but will change in the future to better reflect the semantics). Here we set the lower bound for the <code>scale_min</code> to represent the limits of the backend. If a QConfig's min scale value falls below this limit, the QConfig will be treated as violating this constraint. Note that <code>scale_max_upper_bound</code> is currently not used, because there is no corresponding mechanism to enforce this on the observer yet.</p>
<h4>Fixed quantization parameters</h4>
<p>For ops with fixed quantization parameters such as <code>torch.nn.Sigmoid</code> or <code>torch.nn.Tanh</code>, the BackendConfig can specify the specific scale and zero point values as constraints on the input and output activations. The user's QConfigs for these ops must use <code>FixedQParamsObserver</code> or <code>FixedQParamsFakeQuantize</code> for their activations with matching scale and zero point values, otherwise these QConfigs will be ignored.</p>