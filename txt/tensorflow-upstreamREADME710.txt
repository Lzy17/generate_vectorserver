<h1>Target Aware Conversion (TAC)</h1>
<p>Different hardwares have different capabilities and restrictions.</p>
<p>TAC is designed to leverage hardwares' capabilities to:</p>
<ul>
<li>Perform device-specific optimizations (such as unsupported ops lowering,
    layout transformations, etc.)</li>
<li>Graph partitioning based on the hardware costs modeling.</li>
<li>It supports general import/export where you can hook your own
    importer/exporter from any format to MLIR and export MLIR to anything.</li>
</ul>
<p>For more details, please checkout the
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir/lite/experimental/tac/README.md#tac-workflow">TAC workflow</a>
section</p>
<h2>How to use</h2>
<p>Once you have a converted TfLite model ready, you can use the following command
to use TAC to optimize for your model:</p>
<p><code>bazel run -c opt //tensorflow/compiler/mlir/lite/experimental/tac:tac-translate -- &lt;PATH_TO_YOUR_MODEL&gt; -o=&lt;OUTPUT_PATH&gt; -device-specs=&lt;HARDWARE_BACKENDS&gt;</code></p>
<p>The devices_specs is a list of the names of the desired hardware backends,
separated by comma, e.g., "GPU,CPU".</p>
<p>If you're interested in what are the subgraphs being explored for different
backends, you can pass in <code>-output-mlir -inline-subgraphs=false</code> and check out
the output mlir file.</p>
<h2>How to add a hardware backend</h2>
<p>If you want to add a hardware backend for TAC, you can start with the
<code>SimpleHardware</code> interface.</p>
<p>For example:</p>
<p>```
class FooHardware : public SimpleHardware {
 public:
  static constexpr char kId[] = "FOO";</p>
<p>mlir::RewritePatternSet GetTransformations(
      MLIRContext* context) const override {
    mlir::RewritePatternSet patterns;
    // Pick the transformations that we want to perform,
    // We can add other transformations we like here.
    patterns.add<LowerPackIntoConcatReshape, UnrollSplit, UnrollSplitV,
                  PadSlice>(context);
    return patterns;
  }</p>
<p>mlir::TypeID GetTypeId() const override {
    return mlir::TypeID::get<FooHardware>();
  }</p>
<p>// We can specify what ops are not supported here.
  bool IsNotSupportedOp(mlir::Operation* op) const override { return false; }</p>
<p>// This is basically saying how fast are we comparing to CPU.
  // The larger the value the better.
  float AdvantageOverCPU() const override { return 5.0; }
};
```</p>
<p>Then we need to register our hardware like below:</p>
<p>```
std::unique_ptr<TargetHardware> CreateFooHardware() {
  return std::make_unique<FooHardware>();
}</p>
<p>TargetHardwareRegistration<FooHardware> foo_hardware(
    "Target device for FOO", CreateFooHardware);
```</p>
<h3>Advanced user</h3>
<p>For advanced users (e.g., you may already have your own hardware dialect
defined), please just use <code>TargetHardware</code> directly. See the following code
snippet for reference.</p>
<p>```
class MyCustomHardware : public TargetHardware {
 public:
  static constexpr char kId[] = "MY_CUSTOM_HARDWARE";</p>
<p>mlir::TypeID GetTypeId() const override {
    return mlir::TypeID::get<MyCustomHardware>();
  }</p>
<p>bool IsOpSupported(mlir::Operation* op) const override {
    // check whether the op is supported, if the user has they own dialect,
    // this can be target dialect legalization process.
  }</p>
<p>double GetHardwareSwitchingCost(const TargetHardware* from,
                                 size_t buffer_size) const override {
    // Get the hardware switching cost from the source hardware.
 }</p>
<p>double GetOpCost(mlir::Operation* op) const override {
    // call customized cost model.
  }</p>
<p>mlir::RewritePatternSet GetTransformations(
      MLIRContext* context) const override {
    // customized transformations patterns: ops lowering/fusion, layout
    // transformation, etc.
  }
};
```</p>
<h2>TAC workflow</h2>
<p>The workflow of target-aware-conversion is as followed:</p>
<p>1 Try to break down the whole graph into several subgraphs based on hardwares'
capabilities. See the diagram below, let's say our desired target backends are
"GPU" and "CPU", and currently "C" is not supported on "GPU", but the rest are
supported by "GPU". So we will end up with 3 subgraphs as shown in the diagram.</p>
<p><img alt="Target Annotation" src="g3doc/images/target_annotation.png" /></p>
<p>2  Perform ops-lowering &amp; target-specific optimizations for
    different hardware backends. As shown in the below diagram, the red &amp; the
    yellow subgraph will be duplicated as "alternative subgraph view" for "CPU".
    "C" op can be lowered into "G" + "H" op which can be supported by "GPU".</p>
<p><img alt="Target Optimization" src="g3doc/images/target_optimization.png" /></p>
<p>3  Estimate the costs for each subgraph (and their alternative views)
    based on the hardware cost model. See the following diagram.</p>
<p><img alt="Estimate costs" src="g3doc/images/compute_cost.png" /></p>
<p>4 Pick the proper subgraphs from the alternative views for execution based on
costs(computation costs, transfer costs, quant/dequant costs). As shown in the
diagram below, since cross-device data transferring cost is high, even "G" + "H"
running on GPU maybe less efficient than "C" running on "CPU", we will still
pick "G" + "H" subgraph.</p>
<p><img alt="Pick subgraphs" src="g3doc/images/pick_subgraphs.png" /></p>
<p>The final graph looks like below:</p>
<p><img alt="Final graph" src="g3doc/images/final_graph.png" /></p>
<h2>TAC components</h2>
<h3>Hardwares</h3>
<p>Hardwares are used to modeling target device capabilities &amp; also ops cost for
the target devices.</p>
<p>We have already modeled <code>cpu_hardware</code> &amp; <code>gpu_hardware</code> as well as the
<code>nnapi_hardware</code>.</p>
<h3>Passes</h3>
<h4>Target Annotation Pass</h4>
<p>In this pass, every op will be targeted with the user specified targets based on
the device capabilites. For example, If the user specified the desired targets
are "GPU", "CPU", <code>conv2d</code> can run on both "GPU" and "CPU", we will annotate
the op <code>conv2d</code> with "GPU" since it's preferred; <code>pack</code> can only run on "CPU",
so we will annotate the op with "CPU" since "GPU" does not support this op.</p>
<h4>Raise Target Subgraphs Pass</h4>
<p>In this pass, ops will be broken down into subgraph. Those ops have the same
target annotation will be raised as subgraphs.</p>
<p>In this pass, subgraph is actually implemented with <code>FuncOp</code>.</p>
<p>Take the following code as an example:</p>
<p><code>func @simpleTest(%arg0: tensor&lt;1xf32&gt;, %arg1: tensor&lt;1xf32&gt;, %arg2: tensor&lt;1xf32&gt;, %arg3: tensor&lt;1xf32&gt;) -&gt; tensor&lt;2x1xf32&gt; {
  %0 = "tfl.add"(%arg0, %arg1) {tac.device = "GPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor&lt;1xf32&gt;, tensor&lt;1xf32&gt;) -&gt; tensor&lt;1xf32&gt;
  %1 = "tfl.mul"(%0, %arg2) {tac.device = "GPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor&lt;1xf32&gt;, tensor&lt;1xf32&gt;) -&gt; tensor&lt;1xf32&gt;
  %2 = "tfl.add"(%arg0, %arg3) {tac.device = "GPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor&lt;1xf32&gt;, tensor&lt;1xf32&gt;) -&gt; tensor&lt;1xf32&gt;
  %3 = "tfl.pack"(%1, %2) {tac.device = "CPU", tac.inference_type = "FLOAT", axis = 0 : i32, values_count = 2 : i32} : (tensor&lt;1xf32&gt;, tensor&lt;1xf32&gt;) -&gt; tensor&lt;2x1xf32&gt;
  return %3 : tensor&lt;2x1xf32&gt;
}</code></p>
<p>In this code, <code>%3</code> is annotated with "CPU", while others are annotated with
"GPU", in this case, <code>%3</code> will be raised as a separate function like below:</p>
<p><code>func private @func_1_GPU_FLOAT(%arg0: tensor&lt;1xf32&gt;, %arg1: tensor&lt;1xf32&gt;) -&gt; tensor&lt;1xf32&gt; attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor&lt;1xf32&gt;
    return %0 : tensor&lt;1xf32&gt;
  }</code></p>
<p>And the rest ops will be raised as below:</p>
<p>```
 func private @func_2_CPU_FLOAT(%arg0: tensor&lt;1xf32&gt;, %arg1: tensor&lt;1xf32&gt;) -&gt; tensor&lt;2x1xf32&gt; attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
    %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, tac.device = "CPU", tac.inference_type = "FLOAT", values_count = 2 : i32} : (tensor&lt;1xf32&gt;, tensor&lt;1xf32&gt;) -&gt; tensor&lt;2x1xf32&gt;
    return %0 : tensor&lt;2x1xf32&gt;
  }</p>
<p>func private @func_0_GPU_FLOAT(%arg0: tensor&lt;1xf32&gt;, %arg1: tensor&lt;1xf32&gt;, %arg2: tensor&lt;1xf32&gt;) -&gt; tensor&lt;1xf32&gt; attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor&lt;1xf32&gt;
    %1 = tfl.mul %0, %arg2 {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor&lt;1xf32&gt;
    return %1 : tensor&lt;1xf32&gt;
  }
```</p>
<p>And the original function will be replaced by <code>CallOps</code> to those <code>FuncOps</code>:</p>
<p><code>func @simpleTest(%arg0: tensor&lt;1xf32&gt;, %arg1: tensor&lt;1xf32&gt;, %arg2: tensor&lt;1xf32&gt;, %arg3: tensor&lt;1xf32&gt;) -&gt; tensor&lt;2x1xf32&gt; {
    %0 = call @func_0_GPU_FLOAT(%arg0, %arg1, %arg2) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor&lt;1xf32&gt;, tensor&lt;1xf32&gt;, tensor&lt;1xf32&gt;) -&gt; tensor&lt;1xf32&gt;
    %1 = call @func_1_GPU_FLOAT(%arg0, %arg3) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor&lt;1xf32&gt;, tensor&lt;1xf32&gt;) -&gt; tensor&lt;1xf32&gt;
    %2 = call @func_2_CPU_FLOAT(%0, %1) {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} : (tensor&lt;1xf32&gt;, tensor&lt;1xf32&gt;) -&gt; tensor&lt;2x1xf32&gt;
    return %2 : tensor&lt;2x1xf32&gt;
  }</code></p>
<p>Why we need to raise those ops into <code>FuncOps</code>? Please see the following section.</p>
<h4>Get Alternative Subgraph View Pass</h4>
<p>In the Get Alternative Subgraph View Pass, we will essentially duplicate those
<code>FuncOps</code> and perform unsupported ops lowering &amp; target-specific optimization.</p>
<p>For example, <code>Pack</code> is not supported by "GPU", but it can be lowered into
<code>Concat</code> + <code>Reshape</code> which can be supported by "GPU".</p>
<p>So the original example:</p>
<p><code>func private @func_1_GPU_FLOAT(%arg0: tensor&lt;1xf32&gt;, %arg1: tensor&lt;1xf32&gt;) -&gt; tensor&lt;1xf32&gt; attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor&lt;1xf32&gt;
    return %0 : tensor&lt;1xf32&gt;
  }</code></p>
<p>Will be transformed into:</p>
<p>```
 func private @func_2_CPU_FLOAT(%arg0: tensor&lt;1xf32&gt;, %arg1: tensor&lt;1xf32&gt;) -&gt; tensor&lt;2x1xf32&gt; attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
    %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, tac.device = "CPU", tac.inference_type = "FLOAT", values_count = 2 : i32} : (tensor&lt;1xf32&gt;, tensor&lt;1xf32&gt;) -&gt; tensor&lt;2x1xf32&gt;
    return %0 : tensor&lt;2x1xf32&gt;
  }</p>
<p>func private @func_2_GPU_FLOAT(%arg0: tensor&lt;1xf32&gt;, %arg1: tensor&lt;1xf32&gt;) -&gt; tensor&lt;2x1xf32&gt; attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
    %cst = arith.constant dense&lt;1&gt; : tensor&lt;4xi32&gt;
    %cst_0 = arith.constant dense&lt;2&gt; : tensor&lt;1xi32&gt;
    %cst_1 = arith.constant dense&lt;[2, 1]&gt; : tensor&lt;2xi32&gt;
    %0 = "tfl.reshape"(%arg0, %cst) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor&lt;1xf32&gt;, tensor&lt;4xi32&gt;) -&gt; tensor&lt;1x1x1x1xf32&gt;
    %1 = "tfl.reshape"(%arg1, %cst) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor&lt;1xf32&gt;, tensor&lt;4xi32&gt;) -&gt; tensor&lt;1x1x1x1xf32&gt;
    %2 = "tfl.concatenation"(%0, %1) {axis = 3 : i32, fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor&lt;1x1x1x1xf32&gt;, tensor&lt;1x1x1x1xf32&gt;) -&gt; tensor&lt;1x1x1x2xf32&gt;
    %3 = "tfl.reshape"(%2, %cst_0) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor&lt;1x1x1x2xf32&gt;, tensor&lt;1xi32&gt;) -&gt; tensor&lt;2xf32&gt;
    %4 = "tfl.reshape"(%3, %cst_1) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor&lt;2xf32&gt;, tensor&lt;2xi32&gt;) -&gt; tensor&lt;2x1xf32&gt;
    return %4 : tensor&lt;2x1xf32&gt;
  }
```</p>
<h4>Compute Costs Pass</h4>
<p>In the compute cost pass, we will essentially compute the cost of each op within
the <code>FuncOp</code> based on the target-device cost model and sum them together.</p>
<h4>Pick Subgraphs Pass</h4>
<p>In the pick subgraphs pass, we will pick those subgraphs which can minimize the
global costs (we will take the tensor transferring costs as well).</p>