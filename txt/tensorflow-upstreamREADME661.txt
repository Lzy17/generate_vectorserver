<h1>TensorFlow Lite Sample Stable Delegate</h1>
<h2>Description</h2>
<p>An example delegate for stable delegate testing that supports addition and
subtraction operations only.</p>
<p>The sample stable delegate implementation uses the stable delegate API, which is
based around <code>TfLiteOpaqueDelegate</code>. <code>TfLiteOpaqueDelegate</code> is an opaque version
of <code>TfLiteDelegate</code>; which allows delegation of nodes to alternative backends.
This is an abstract type that is intended to have the same role as
<code>TfLiteDelegate</code> from
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h">common.h</a>,
but without exposing the implementation details of how delegates are
implemented.</p>
<p><code>TfLiteOpaqueDelegate</code>s can be loaded dynamically (see
<code>sample_stable_delegate_external_test.cc</code>) and then be supplied to the TFLite
runtime, in the same way as statically linked delegates can.</p>
<p>Note however that open-source TF Lite does not (yet) provide a binary stable
interface between delegates and the TF Lite runtime itself. Therefore any opaque
delegate that is loaded dynamically into TF Lite <em>must</em> have been built against
the same version (and commit) that the TF Lite runtime itself has been built at.
Any other configuration can lead to undefined behavior.</p>
<h2>Delegate implementation</h2>
<p>The sample stable delegate uses two supporting interfaces
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_opaque_delegate.h">SimpleOpaqueDelegateInterface and SimpleOpaqueDelegateKernelInterface</a>.
These APIs make it easier to implement an opaque TF Lite delegate, though their
usage is entirely optional.</p>
<p>The <code>sample_stable_delegate_test</code> driver (see next section) makes use of the
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_opaque_delegate.h">TfLiteOpaqueDelegateFactory</a>
facility, which provides static methods that deal with delegate creation and
deletion.</p>
<h2>Testing</h2>
<p>See
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate_test.cc">sample_stable_delegate_test.cc</a>
for a standalone test driver that links the sample stable delegate statically
and runs inference on a TF Lite model.</p>
<p>See
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate_external_test.cc">sample_stable_delegate_external_test.cc</a>
for a standalone test driver that loads the sample stable delegate dynamically
and runs inference on a TF Lite model.</p>
<h3>Delegate Test Suite</h3>
<p>The Delegate Test Suite provides correctness testing for a delegate at the
operation level. It checks whether a delegate produces results that meet the
accuracy thresholds of the supported operations.</p>
<p>Support for stable delegate binaries has been integrated into the Delegate Test
Suite.</p>
<h4>Run on Android</h4>
<p>The following instructions show how to run the test suite on Android.</p>
<p>First, we build the sample stable delegate shared library file,
<code>libtensorflowlite_sample_stable_delegate.so</code>, which we will later load
dynamically as part of the test:</p>
<p>```bash
bazel build -c opt --config=android_arm64 //tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate</p>
<p>adb push "$(bazel info -c opt --config=android_arm64 bazel-bin)"/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so /data/local/tmp
```</p>
<p>Next, we create a configuration file for the component that loads the stable
delegate:</p>
<p><code>bash
adb shell 'echo "{
  \"stable_delegate_loader_settings\": {
    \"delegate_path\": \"/data/local/tmp/libtensorflowlite_sample_stable_delegate.so\"
  }
  // Add concrete delegate settings for the test target delegate.
}
"&gt; /data/local/tmp/stable_delegate_settings.json'</code></p>
<p>We create a configuration file for the delegate test suite to verify that the
models in the specified test cases have been delegated:</p>
<p><code>bash
adb shell 'echo "
  # The sample stable delegate supports static-sized addition and subtraction operations.
  FloatSubOpModel.NoActivation
  FloatSubOpModel.VariousInputShapes
  FloatAddOpModel.NoActivation
  FloatAddOpModel.VariousInputShapes
"&gt; /data/local/tmp/stable_delegate_acceleration_test_config.json'</code></p>
<p>Then, we build the test suite itself:</p>
<p>```bash
bazel build -c opt --config=android_arm64 //tensorflow/lite/delegates/utils/experimental/stable_delegate:stable_delegate_test_suite</p>
<p>adb push "$(bazel info -c opt --config=android_arm64 bazel-bin)"/tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_test_suite /data/local/tmp
```</p>
<p>Now, we can execute the test suite with providing the settings file:</p>
<p><code>bash
adb shell "/data/local/tmp/stable_delegate_test_suite \
  --stable_delegate_settings_file=/data/local/tmp/stable_delegate_settings.json \
  --acceleration_test_config_path=/data/local/tmp/stable_delegate_acceleration_test_config.json"</code></p>
<p>The test suite will show the following output in console after all tests are
passed:</p>
<p><code>...
[==========] 3338 tests from 349 test suites ran. (24555 ms total)
[  PASSED  ] 3338 tests.</code></p>
<h3>Benchmark Tools</h3>
<h4>Delegate Performance Benchmark app</h4>
<p>The
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/README.md">Delegate Performance Benchmark app</a>
is the recommended tool to test the latency and accuracy of a stable delegate.</p>
<h4>TF Lite Benchmark Tool</h4>
<p>During early development stages of a new stable delegate it can also be useful
to directly load the delegate's shared library file into TF Lite's
<code>benchmark_model</code> tool, because this development workflow works on regular linux
desktop machines and also allows users to benchmark any TF Lite model file they
are interested in.</p>
<p>Support for stable delegate binaries has been integrated into TF Lite's
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark"><code>benchmark_model</code></a>
CLI tool. We can use this tool to test the sample stable delegate with a
provided TF Lite model file.</p>
<h5>A) Run on a regular linux host</h5>
<p>The following instructions show how to run the tool on regular desktop linux
machine.</p>
<p>First, we build the sample stable delegate shared library file,
<code>libtensorflowlite_sample_stable_delegate.so</code>, which we will later load
dynamically with the <code>benchmark_model</code> tool:</p>
<p><code>bash
bazel build -c opt //tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate</code></p>
<p>Next, we create a configuration file for the component that loads the stable
delegate:</p>
<p><code>bash
echo "{
  \"stable_delegate_loader_settings\": {
    \"delegate_path\": \"$(bazel info -c opt bazel-bin)/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so\"
  }
  // Add concrete delegate settings for the test target delegate.
}
"&gt; stable_delegate_settings.json</code></p>
<p>Then, we build the <code>benchmark_model</code> tool itself:</p>
<p><code>bash
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model</code></p>
<p>Now, we can execute the benchmark tool. We provide the settings file together
with a TF Lite file that contains ADD operations. We do this because the sample
stable delegate only support ADD and SUB:</p>
<p><code>bash
$(bazel info -c opt bazel-bin)/tensorflow/lite/tools/benchmark/benchmark_model \
  --stable_delegate_settings_file=$(pwd)/stable_delegate_settings.json \
    --graph=$(pwd)/tensorflow/lite/testdata/add.bin</code></p>
<p>Note that when you make changes to the sample delegate you need to rebuild the
delegate's shared library file, in order for benchmark_model to pick up the new
delegate code.</p>
<h5>B) Run on Android</h5>
<p>The following instructions show how to run the tool on Android.</p>
<p>First, we build the sample stable delegate shared library file,
<code>libtensorflowlite_sample_stable_delegate.so</code>, which we will later load
dynamically with the <code>benchmark_model</code> tool:</p>
<p>```bash
bazel build -c opt --config=android_arm64 //tensorflow/lite/delegates/utils/experimental/sample_stable_delegate:tensorflowlite_sample_stable_delegate</p>
<p>adb push "$(bazel info -c opt --config=android_arm64 bazel-bin)"/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so /data/local/tmp
```</p>
<p>Next, we create a configuration file for the component that loads the stable
delegate:</p>
<p><code>bash
adb shell 'echo "{
  \"stable_delegate_loader_settings\": {
    \"delegate_path\": \"/data/local/tmp/libtensorflowlite_sample_stable_delegate.so\"
  }
  // Add concrete delegate settings for the test target delegate.
}
"&gt; /data/local/tmp/stable_delegate_settings.json'</code></p>
<p>Then, we build the <code>benchmark_model</code> tool itself:</p>
<p>```bash
bazel build -c opt --config=android_arm64 //tensorflow/lite/tools/benchmark:benchmark_model</p>
<p>adb push "$(bazel info -c opt --config=android_arm64 bazel-bin)"/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp
```</p>
<p>Now, we can execute the benchmark tool. We provide the settings file together
with a TF Lite file that contains ADD operations. We do this because the sample
stable delegate only support ADD and SUB:</p>
<p><code>bash
adb push tensorflow/lite/testdata/add.bin /data/local/tmp/add.bin
adb shell "/data/local/tmp/benchmark_model \
  --stable_delegate_settings_file=/data/local/tmp/stable_delegate_settings.json \
  --graph=/data/local/tmp/add.bin"</code></p>