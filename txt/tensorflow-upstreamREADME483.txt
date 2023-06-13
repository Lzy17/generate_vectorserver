<p>Image classification using the ResNet50 model described in
<a href="https://arxiv.org/abs/1512.03385">Deep Residual Learning for Image Recognition</a>.</p>
<p>Contents:</p>
<ul>
<li><code>resnet50.py</code>: Model definition</li>
<li><code>resnet50_test.py</code>: Sanity unittests and benchmarks for using the model with
  eager execution enabled.</li>
<li><code>resnet50_graph_test.py</code>: Sanity unittests and benchmarks when using the same
  model code to construct a TensorFlow graph.</li>
</ul>
<h1>Benchmarks</h1>
<p>Using a synthetic data, run:</p>
<p>```</p>
<h1>Using eager execution</h1>
<p>python resnet50_test.py --benchmark_filter=.</p>
<h1>Using graph execution</h1>
<p>python resnet50_graph_test.py --benchmark_filter=.
```</p>
<p>The above uses the model definition included with the TensorFlow pip
package. To build (and run benchmarks) from source:</p>
<p>```</p>
<h1>Using eager execution</h1>
<p>bazel run -c opt --config=cuda :resnet50_test -- --benchmark_filter=.</p>
<h1>Using graph execution</h1>
<p>bazel run -c opt --config=cuda :resnet50_graph_test -- --benchmark_filter=.
```</p>
<p>(Or remove the <code>--config=cuda</code> flag for running on CPU instead of GPU).</p>
<p>On October 31, 2017, the benchmarks demonstrated comparable performance
for eager and graph execution of this particular model when using
a single NVIDIA Titan X (Pascal) GPU on a host with an
Intel Xeon E5-1650 CPU @ 3.50GHz and a batch size of 32.</p>
<p>| Benchmark name                           | batch size    | images/second |
| ---------------------------------------  | ------------- | ------------- |
| eager_train_gpu_batch_32_channels_first  |            32 |           171 |
| graph_train_gpu_batch_32_channels_first  |            32 |           172 |</p>