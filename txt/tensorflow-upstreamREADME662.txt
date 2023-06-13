<p>When speaking of a TFLite delegate, how to create it and how to reuse existing
TFLite testing and tooling with the new delegate are two major challenging
issues. Here, we show a dummy delegate implementation to illustrate our
recommended approaches to address these issues.</p>
<h2>Delegate Creation</h2>
<p>We recommend using
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_delegate.h">SimpleDelegateInterface and SimpleDelegateKernelInterface</a>.
We believe such APIs will make it easier to create a TFLite delegate. At a high
level, developers only need to focus on</p>
<ul>
<li>Whether a TFLite node in the graph is supported by the delegate or not.</li>
<li>Given the set of supported nodes (i.e. a subgraph of the original model
graph), implement a delegate kernel that executes this set of nodes.</li>
</ul>
<p>The dummy delegate implementation here is a good starting point to understand
the ideas above. For more sophisticated examples, refer to <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/flex">Flex delegate</a>,
    <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/hexagon">Hexagon delegate</a>.</p>
<h2>Testing &amp; Tooling</h2>
<p>There are currently <strong>two options</strong> to plug in a newly created TFLite delegate
to reuse existing TFLite kernel tests and tooling:</p>
<ul>
<li>Utilize the <strong><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates">delegate registrar</a></strong>
mechanism</li>
<li>Utilize the
<strong><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/external">external delegate</a></strong>
mechanism.</li>
</ul>
<p>The former approach requires few changes as detailed below. The latter one
requires even fewer changes and works with pre-built Tensorflow Lite tooling
binaries. However, it is less explicit and it might be more complicated to set
up in automated integration tests. Therefore, for better clarity, the
delegate-registrar approach is slightly preferred here.</p>
<p>We now describe each option above in more details in the following sections.</p>
<h3>Option 1: Utilize Delegate Registrar</h3>
<p>In this approach, create a delegate provider like the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate_provider.cc"><code>dummy_delegate_provider.cc</code></a>
here, and then add it as an extra dependency when building the binary. Refer
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates">here</a>
for more delegate provider examples. Now we look at using this provider for
testing and evaluation.</p>
<h4>Kernel Tests</h4>
<p>Tests referred here are defined in <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels">tensorflow/lite/kernels</a>.
They are based on the
 <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/test_util.h">test_util library</a>
 and the <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/test_main.cc">testing main function stub</a>.</p>
<p>To plug in the newly created delegate and reuse these tests, simply add the
created delegate provider as an extra dependency to
<a href="https://github.com/tensorflow/tensorflow/blob/f09dc5cf6e7fde978f9891638f529cd52a3c878f/tensorflow/lite/kernels/BUILD#L203"><code>test_util_delegate_providers</code></a>
and remove others that are not relevant, like the following:</p>
<p><code>cc_library(
    name = "tflite_driver_delegate_providers",
    deps = [
        # Existing delegate providers that might be still relevant.
        ":dummy_delegate_provider",
    ],
    alwayslink = 1,
)</code></p>
<p>Then build a kernel test, and specify the commandline flags defined in the
delegate provider when executing the test. Take this case as an example,</p>
<p>```
bazel build -c opt tensorflow/lite/kernels:add_test</p>
<h1>Setting --use_dummy_delegate=true will apply the dummy delegate to the</h1>
<h1>TFLite model graph</h1>
<p>bazel-bin/tensorflow/lite/kernels/add_test --use_dummy_delegate=true
```</p>
<h4>Benchmark and Task Evaluation Tools</h4>
<p>In TFLite, we have developed
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark">model benchmark tool</a>
and
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks">evaluation tools</a>
that already have integrated existing various TFLite delegates. To reuse these
tools for the new delegate, similar to the kernel testing above, we simply add
the created delegate provider as an additional dependency when building the
binary. See rules in the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/BUILD">BUILD</a>
file for details.</p>
<p>Take reusing the TFLite model benchmark tool as an example, after the delegate
provider is created, define the BUILD rule like the following:</p>
<p><code>cc_binary(
    name = "benchmark_model_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        # Simply add the delegate provider as an extra dep.
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
    ],
)</code></p>
<p>Now build the binary, and specify the commandline flags defined in this new
delegate provider and others detailed in the benchmark model tool
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/README.md">doc</a>
when running the benchmark tool like the following:</p>
<p>```
bazel build -c opt tensorflow/lite/delegates/utils/dummy_delegate:benchmark_model_plus_dummy_delegate</p>
<h1>Setting --use_dummy_delegate=true will apply the dummy delegate to the</h1>
<h1>TFLite model graph.</h1>
<p>bazel-bin/tensorflow/lite/delegates/utils/dummy_delegate/benchmark_model_plus_dummy_delegate --graph=/tmp/mobilenet-v2.tflite --use_dummy_delegate=true</p>
<p>```</p>
<h3>Option 2: Utilize Tensorflow Lite External Delegate</h3>
<p>In this <strong>alternative approach to reuse existing Tensorflow Lite kernel testing
and tooling</strong>, we first create an external delegate adaptor like the <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/dummy_delegate/external_delegate_adaptor.cc"><code>external_delegate_adaptor.cc</code></a> here, and create the corresponding BUILD target
to build a dynamic library.</p>
<p>Afterwards, one could build binaries or use pre-built ones to run with the
dummy delegate as long as the binary is linked with the
<a href="https://github.com/tensorflow/tensorflow/blob/8c6f2d55762f3fc94f98fdd8b3c5d59ee1276dba/tensorflow/lite/tools/delegates/BUILD#L145-L159"><code>external_delegate_provider</code></a>
library which supports command-line flags as described
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/delegates#external-delegate-provider">here</a>.
Note this external delegate provider has already been linked to existing testing
and tooling binaries.</p>
<p>For example, the following illustrates how to benchmark the dummy delegate here
via this external-delegate approach. We could use similar commands for testing
and evaluation tools.</p>
<p>```
bazel build -c opt tensorflow/lite/delegates/utils/dummy_delegate:dummy_external_delegate.so</p>
<h1>Copy the .so file to the directory that the external delegate will be loaded</h1>
<h1>from at your choice.</h1>
<p>cp bazel-bin/tensorflow/lite/delegates/utils/dummy_delegate/dummy_external_delegate.so /tmp</p>
<p>bazel build -c opt tensorflow/lite/tools/benchmark:benchmark_model</p>
<h1>Setting a non-empty --external_delegate_path value will trigger applying</h1>
<h1>the external delegate during runtime.</h1>
<p>bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=/tmp/mobilenet-v2.tflite \
  --external_delegate_path=/tmp/dummy_external_delegate.so \
  --external_delegate_options='error_during_init:true;error_during_prepare:true'
```</p>
<p>It is worth noting the <em>external delegate</em> is the corresponding C++
implementation of the <em>delegate</em> in Tensorflow Lite Python binding as shown
<a href="https://github.com/tensorflow/tensorflow/blob/7145fc0e49be01ef6943f4df386ce38567e37797/tensorflow/lite/python/interpreter.py#L42">here</a>.
Therefore, the dynamic external delegate adaptor library created here could be
directly used with Tensorflow Lite Python APIs.</p>
<p>More detailed guide on TFLite delegate is coming soon.</p>