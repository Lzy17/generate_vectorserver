<h1>TensorFlow Lite Quantization Debugger</h1>
<p>[TOC]</p>
<h2>Overview</h2>
<p>When a quantized model is produced, it requires tedious and manual custom code
to debug the model in order to:</p>
<ol>
<li>Verify if the quantized model is working as expected (spot errors, check
   accuracy, etc).</li>
<li>Compare the quantized model and the original float model.</li>
</ol>
<p>This is now feasible using the TensorFlow Lite Quantization Debugger, as shown
below.</p>
<p>Note: Currently, this workflow is only supported for full integer (int8)
quantization. The debug model produced using this workflow should only be used
for debugging purposes only (and not for inference).</p>
<h2>Analysis with quantized model only</h2>
<h3>Produce a debug model</h3>
<p>Modify the
<a href="https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization">TFLite full integer (int8) quantization steps</a>
as shown below to produce a debug model (used for debugging purposes only, and
not inference)</p>
<h4>How does this work?</h4>
<p>With the help of the MLIR quantizer's debug mode feature, the debug model
produced has both the original float operators (or ops) and the quantized ops.
Additionally, <code>NumericVerify</code> ops are added to compare the outputs of the
original float and quantized ops and to also collect statistics. It has the name
in the format of <code>NumericVerify/{original tensor name}:{original tensor id}</code></p>
<p>```python</p>
<h1>for mlir_quantize</h1>
<p>from tensorflow.lite.python import convert</p>
<h1>set full-integer quantization parameters as usual.</h1>
<p>converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = calibration_gen</p>
<h1>Create a TFLite model with new quantizer and numeric verify ops. Rather than</h1>
<h1>calling convert() only, calibrate model first and call <code>mlir_quantize</code> to run</h1>
<h1>the actual quantization, with <code>enable_numeric_verify</code> set to <code>True</code>.</h1>
<p>converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter._experimental_calibrate_only = True
calibrated = converter.convert()
return convert.mlir_quantize(calibrated, enable_numeric_verify=True)
```</p>
<h3>Run debugger with debug model</h3>
<p>Initialize debugger with the debug model. This can be done in two ways.</p>
<p>```python
from tensorflow.lite.tools.optimize.debugging.python import debugger</p>
<h1><code>debug_dataset</code> accpets the same type as <code>converter.representative_dataset</code>.</h1>
<p>quant_debugger = debugger.QuantizationDebugger(
    quant_debug_model_content=quant_debug_model,
    debug_dataset=data_gen)</p>
<h1>OR</h1>
<p>quant_debugger = debugger.QuantizationDebugger(
    quant_debug_model_path='/path/to/debug_model.tflite',
    debug_dataset=data_gen)</p>
<p>quant_debugger.run()
```</p>
<h3>Inspect statistics</h3>
<p>When you call <code>quant_debugger.run()</code>, <code>quant_debugger.layer_statistics</code> is
filled with aggregated statistics for each <code>NumericVerify</code> ops. Some metrics
(i.e. stddev, mean square error) are calculated by default.</p>
<h4>Example output</h4>
<p>```python</p>
<h1><code>quant_debugger.layer_statistics.metrics</code> is defaultdict, convert it to dict</h1>
<h1>for readable output.</h1>
<p>import pprint
for layer_name, metrics in quant_debugger.layer_statistics.items():
  print(layer_name)
  pprint.pprint(dict(metrics))
```</p>
<p>```python</p>
<h1>...</h1>
<p>NumericVerify/sequential/dense/MatMul;sequential/dense/BiasAdd3:77
{'max_abs_error': 0.05089309,
 'mean_error': -0.00017149668,
 'mean_squared_error': 0.00040816222,
 'num_elements': 256.0,
 'stddev': 0.02009948}
NumericVerify/sequential/dense_1/MatMul;sequential/dense_1/BiasAdd3:81
{'max_abs_error': 0.09744112,
 'mean_error': 0.0048679365,
 'mean_squared_error': 0.0036721828,
 'num_elements': 10.0,
 'stddev': 0.055745363}
NumericVerify/Identity2:85
{'max_abs_error': 0.0036417267,
 'mean_error': -0.00068773015,
 'mean_squared_error': 3.439951e-06,
 'num_elements': 10.0,
 'stddev': 0.0016223773}</p>
<h1>...</h1>
<p>```</p>
<h2>Adding custom metrics</h2>
<p>More metrics can be added by passing <code>QuantizationDebugOptions</code> to the
initializer. For example, if you want to add mean absolute error, use following
snippet.</p>
<p>```python
debug_options = debugger.QuantizationDebugOptions(
    layer_debug_metrics={
        'mean_abs_error': lambda diffs: np.mean(np.abs(diffs))
    })</p>
<p>quant_debugger = debugger.QuantizationDebugger(
    quant_debug_model_content=quant_debug_model,
    debug_dataset=data_gen,
    debug_options=debug_options
)
quant_debugger.run()
```</p>
<p>Now <code>quant_debugger.layer_statistics</code> includes mean absoulte error for each
layer.</p>
<h2>Analysis with float and quantized models</h2>
<p>In addition to single model analysis, the output of original float model and
quantized model can be compared when both models are given. This can be done
by providing a float model, and metrics to compare outputs. This can be <code>argmax</code>
for classification models, bit for more complex models like detection more
complicated logic should be given.</p>
<p>```python</p>
<h1>functions for model_debug_metrics gets all output tensors from float and</h1>
<h1>quantized models, and returns a single metric value.</h1>
<p>debug_options = debugger.QuantizationDebugOptions(
    model_debug_metrics={
        'argmax_accuracy': lambda f, q: np.argmax(f[0]) == np.argmax(q[0])
    })</p>
<p>float_model = converter.convert()  # converted without any optimizations.</p>
<p>quant_debugger = debugger.QuantizationDebugger(
    quant_debug_model_content=quant_debug_model,
    float_model_content=float_model,  # can pass <code>float_model_path</code> instead.
    debug_dataset=data_gen,
    debug_options=debug_options
)
quant_debugger.run()
```</p>
<p>The result is a single number per metric, so it's easier to inspect.</p>
<p>```python</p>
<blockquote>
<blockquote>
<blockquote>
<p>quant_debugger.model_statistics
{'argmax_accuracy': 0.89}
```</p>
</blockquote>
</blockquote>
</blockquote>
<h2>Advanced usage: Export stats to csv, and import to pandas</h2>
<p><code>quant_debugger.layer_statistics_dump</code> function accepts file-like object, and
exports layer statistics to csv. This can be imported to other tools like
<code>pandas</code> for further processing. The exported data also has name of the op,
originating tensor ID, and quantization parameters (scales and zero points) for
quantized layer.</p>
<p>Note: scales and zero points are lists, and imported to <code>pandas</code> as text by
default. Additional processing to parse them is required before processing.</p>
<p>```python
import pandas as pd
import yaml  # used to parse lists</p>
<p>with open('/path/to/stats.csv', 'w') as f:
  quant_debugger.layer_statistics_dump(f)</p>
<p>data = pd.read_csv(
    '/path/to/stats.csv',
    converters={
        'scales': yaml.safe_load,
        'zero_points': yaml.safe_load
    })
```</p>