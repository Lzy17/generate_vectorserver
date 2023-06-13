<h1>TFLite Delegate Utilities for Tooling</h1>
<h2>TFLite Delegate Registrar</h2>
<p><a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/delegate_provider.h">A TFLite delegate registrar</a>
is provided here. The registrar keeps a list of TFLite delegate providers, each
of which defines a list parameters that could be initialized from commandline
arguments and provides a TFLite delegate instance creation based on those
parameters. This delegate registrar has been used in TFLite evaluation tools and
the benchmark model tool.</p>
<p>A particular TFLite delegate provider can be used by linking the corresponding
library, e.g. adding it to the <code>deps</code> of a BUILD rule. Note that each delegate
provider library has been configured with <code>alwayslink=1</code> in the BUILD rule so
that it will be linked to any binary that directly or indirectly depends on it.</p>
<p>The following lists all implemented TFLite delegate providers and their
corresponding list of parameters that each supports to create a particular
TFLite delegate.</p>
<h3>Common parameters</h3>
<ul>
<li><code>num_threads</code>: <code>int</code> (default=-1) \
    The number of threads to use for running the inference on CPU. By default,
    this is set to the platform default value -1.</li>
<li><code>max_delegated_partitions</code>: <code>int</code> (default=0, i.e. no limit) \
    The maximum number of partitions that will be delegated. \
    Currently supported by the GPU, Hexagon, CoreML and NNAPI delegate.</li>
<li><code>min_nodes_per_partition</code>: <code>int</code> (default=delegate's own choice) \
    The minimal number of TFLite graph nodes of a partition that needs to be
    reached to be delegated. A negative value or 0 means to use the default
    choice of each delegate. \
    This option is currently supported by the Hexagon and CoreML delegate.</li>
<li><code>delegate_serialize_dir</code>: <code>string</code> (default="") \
    Directory to be used by delegates for serializing any model data. This
    allows the delegate to save data into this directory to reduce init time
    after the first run. Currently supported by GPU (OpenCL) and NNAPI delegate
    with specific backends on Android. Note that delegate_serialize_token is
    also required to enable this feature.</li>
<li><code>delegate_serialize_token</code>: <code>string</code> (default="") \
    Model-specific token acting as a namespace for delegate serialization.
    Unique tokens ensure that the delegate doesn't read inapplicable/invalid
    data. Note that delegate_serialize_dir is also required to enable this
    feature.</li>
<li><code>first_delegate_node_index</code>: <code>int</code> (default=0) \
    The index of the first node that could be delegated. Debug only. Add
    '--define=tflite_debug_delegate=true' in your build command line to use it.
    \
    Currently only supported by CoreML delegate.</li>
<li><code>last_delegate_node_index</code>: <code>int</code> (default=INT_MAX) \
    The index of the last node that could be delegated. Debug only. Add
    '--define=tflite_debug_delegate=true' in your build command line to use it.
    \
    Currently only supported by CoreML delegate.</li>
</ul>
<h3>GPU delegate provider</h3>
<p>The GPU delegate is supported on Android and iOS devices, or platforms where the
delegate library is built with "-DCL_DELEGATE_NO_GL" macro.</p>
<h4>Common options</h4>
<ul>
<li><code>use_gpu</code>: <code>bool</code> (default=false) \
    Whether to use the
    <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu">GPU accelerator delegate</a>.</li>
<li><code>gpu_precision_loss_allowed</code>: <code>bool</code> (default=true) \
    Whether to allow the GPU delegate to carry out computation with some
    precision loss (i.e. processing in FP16) or not. If allowed, the performance
    will increase.</li>
<li><code>gpu_experimental_enable_quant</code>: <code>bool</code> (default=true) \
    Whether to allow the GPU delegate to run a 8-bit quantized model or not.</li>
<li><code>gpu_inference_for_sustained_speed</code>: <code>bool</code> (default=false) \
    Whether to prefer maximizing the throughput. This mode will help when the
    same delegate will be used repeatedly on multiple inputs. This is supported
    on non-iOS platforms.</li>
</ul>
<h4>Android options</h4>
<ul>
<li><code>gpu_backend</code>: <code>string</code> (default="") \
    Force the GPU delegate to use a particular backend for execution, and fail
    if unsuccessful. Should be one of: cl, gl. By default, the GPU delegate will
    try OpenCL first and then OpenGL if the former fails.</li>
</ul>
<h4>iOS options</h4>
<ul>
<li><code>gpu_wait_type</code>: <code>string</code> (default="") \
    Which GPU wait_type option to use. Should be one of the following: passive,
    active, do_not_wait, aggressive. When left blank, passive mode is used by
    default.</li>
</ul>
<h3>NNAPI delegate provider</h3>
<ul>
<li><code>use_nnapi</code>: <code>bool</code> (default=false) \
    Whether to use
    <a href="https://developer.android.com/ndk/guides/neuralnetworks/">Android NNAPI</a>.
    This API is available on recent Android devices. When on Android Q+, will
    also print the names of NNAPI accelerators accessible through the
    <code>nnapi_accelerator_name</code> flag.</li>
<li><code>nnapi_accelerator_name</code>: <code>string</code> (default="") \
    The name of the NNAPI accelerator to use (requires Android Q+). If left
    blank, NNAPI will automatically select which of the available accelerators
    to use.</li>
<li><code>nnapi_execution_preference</code>: <code>string</code> (default="") \
    Which
    <a href="https://developer.android.com/ndk/reference/group/neural-networks.html#group___neural_networks_1gga034380829226e2d980b2a7e63c992f18af727c25f1e2d8dcc693c477aef4ea5f5">NNAPI execution preference</a>
    to use when executing using NNAPI. Should be one of the following:
    fast_single_answer, sustained_speed, low_power, undefined.</li>
<li><code>nnapi_execution_priority</code>: <code>string</code> (default="") \
    The relative priority for executions of the model in NNAPI. Should be one of
    the following: default, low, medium and high. This option requires Android
    11+.</li>
<li><code>disable_nnapi_cpu</code>: <code>bool</code> (default=true) \
    Excludes the
    <a href="https://developer.android.com/ndk/guides/neuralnetworks#device-assignment">NNAPI CPU reference implementation</a>
    from the possible devices to be used by NNAPI to execute the model. This
    option is ignored if <code>nnapi_accelerator_name</code> is specified.</li>
<li><code>nnapi_allow_fp16</code>: <code>bool</code> (default=false) \
    Whether to allow FP32 computation to be run in FP16.</li>
<li><code>nnapi_allow_dynamic_dimensions</code>: <code>bool</code> (default=false) \
    Whether to allow dynamic dimension sizes without re-compilation. This
    requires Android 9+.</li>
<li><code>nnapi_use_burst_mode</code>: <code>bool</code> (default=false) \
    use NNAPI Burst mode if supported. Burst mode allows accelerators to
    efficiently manage resources, which would significantly reduce overhead
    especially if the same delegate instance is to be used for multiple
    inferences.</li>
<li><code>nnapi_support_library_path</code>: <code>string</code> (default=""), Path from which NNAPI
    support library will be loaded to construct the delegate. In order to use
    NNAPI delegate with support library, --nnapi_accelerator_name must be
    specified and must be equal to one of the devices provided by the support
    library.</li>
</ul>
<h3>Hexagon delegate provider</h3>
<ul>
<li><code>use_hexagon</code>: <code>bool</code> (default=false) \
    Whether to use the Hexagon delegate. Not all devices may support the Hexagon
    delegate, refer to the
    <a href="https://www.tensorflow.org/lite/performance/hexagon_delegate">TensorFlow Lite documentation</a>
    for more information about which devices/chipsets are supported and about
    how to get the required libraries. To use the Hexagon delegate also build
    the hexagon_nn:libhexagon_interface.so target and copy the library to the
    device. All libraries should be copied to /data/local/tmp on the device.</li>
<li><code>hexagon_profiling</code>: <code>bool</code> (default=false) \
    Whether to profile ops running on hexagon.</li>
</ul>
<h3>XNNPACK delegate provider</h3>
<ul>
<li><code>use_xnnpack</code>: <code>bool</code> (default=false) \
    Whether to explicitly apply the XNNPACK delegate. Note the XNNPACK delegate
    could be implicitly applied by the TF Lite runtime regardless the value of
    this parameter. To disable this implicit application, set the value to
    <code>false</code> explicitly.</li>
</ul>
<h3>CoreML delegate provider</h3>
<ul>
<li><code>use_coreml</code>: <code>bool</code> (default=false) \
    Whether to use the
    <a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/coreml">Core ML delegate</a>.
    This option is only available in iOS.</li>
<li><code>coreml_version</code>: <code>int</code> (default=0) \
    Target Core ML version for model conversion. The default value is 0 and it
    means using the newest version that's available on the device.</li>
</ul>
<h3>External delegate provider</h3>
<ul>
<li><code>external_delegate_path</code>: <code>string</code> (default="") \
    Path to the external delegate library to use.</li>
<li><code>external_delegate_options</code>: <code>string</code> (default="") \
    A list of options to be passed to the external delegate library. Options
    should be in the format of <code>option1:value1;option2:value2;optionN:valueN</code></li>
</ul>
<h3>Stable delegate provider [Experimental API]</h3>
<p>The stable delegate provider provides a <code>TfLiteOpaqueDelegate</code> object pointer
and its corresponding deleter by loading a dynamic library that encapsulates the
actual TFLite delegate implementation in a
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/acceleration/configuration/c/stable_delegate.h"><code>TfLiteStableDelegate</code></a>
struct instance.</p>
<p>While the structure of the stable delegate provider is similar to the external
delegate provider, which provides the
<a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/external">external delegates</a>,
the design objectives of the stable delegates and the external delegates are
different.</p>
<ul>
<li>Stable delegates are designed to work with shared object files that support
    ABI backward compatibility; that is, the delegate and the TF Lite runtime
    won't need to be built using the exact same version of TF Lite as the app.
    However, this is work in progress and the ABI stability is not yet
    guaranteed.</li>
<li>External delegates were developed mainly for delegate evaluation
    (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/external).</li>
</ul>
<p>The stable delegates and the external delegates use different APIs for
diagnosing errors, creating and destroying the delegates. For more details of
the concrete API differences, please check
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/acceleration/configuration/c/stable_delegate.h">stable_delegate.h</a>
and
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/external/external_delegate.h">external_delegate.h</a>.</p>
<p>The stable delegate provider is not supported on Windows platform.</p>
<ul>
<li><code>stable_abi_delegate_settings_file</code>: <code>string</code> (default="") \
    Path to the delegate settings JSON file.</li>
</ul>