<h1>Hexagon Delegate</h1>
<p>Delegate which uses Hexagon SDK to delegate the processing to QC DSP.
Note that we only support quantized models, since the DSP is efficient
with quantized versions. So all op support is for quantized versions.</p>
<p>For more detailed usage and examples check the <a href="https://www.tensorflow.org/lite/performance/hexagon_delegate">user guide.</a></p>
<p>Usage:</p>
<ul>
<li>
<p>Add dependency on hexagon_delegate rule.</p>
</li>
<li>
<p>Code change example:</p>
</li>
</ul>
<p>```
  #include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"</p>
<p>// Assuming shared libraries are under "/data/local/tmp/"
  // If files are packaged with native lib in android App then it
  // will typically be equivalent to the path provided by
  // "getContext().getApplicationInfo().nativeLibraryDir"
  const char[] library_directory_path = "/data/local/tmp/";
  TfLiteHexagonInitWithPath(library_directory_path);  // Needed once at startup.
  ::tflite::TfLiteHexagonDelegateOptions params = {0};
  // 'delegate_ptr' Need to outlive the interpreter. For example,
  // If use case will need to resize input or anything that can trigger
  // re-applying delegates then 'delegate_ptr' need to outlive the interpreter.
  auto* delegate_ptr = ::tflite::TfLiteHexagonDelegateCreate(&amp;params);
  Interpreter::TfLiteDelegatePtr delegate(delegate_ptr,
      <a href="TfLiteDelegate* delegate"></a> {
        ::tflite::TfLiteHexagonDelegateDelete(delegate);
      });
  interpreter-&gt;ModifyGraphWithDelegate(delegate.get());
  TfLiteHexagonTearDown();  // Needed once at end of app/DSP usage.
```</p>
<ul>
<li>Shared libraries:</li>
<li>'libhexagon_interface.so' which holds the interface that the delegate uses.
  It must be available if you linked the hexagon_delegate library to TFLite.
  You can load it either from shell by overriding
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"path to the so",
  or add it inside your apk in a way it is available.</li>
<li>'libhexagon_nn_skel(_v65/_v66).so' which holds the DSP code.
  Use TfLiteHexagonInitWithPath(..) and provide the path to the directory
  which holds the shared libraries for the Hexagon NN on device.
  If you're using TfLiteHexagonInit() then
  You will need to set environment variable "ADSP_LIBRARY_PATH" to
  "path_to_the_lib";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp
  Note that separator here is ';' not ':'
  You can push all 3 files, and the library will pick the one needed based
  on the runtime. Or if you are sure of what you will use on the device then
  push only one of them.</li>
</ul>
<h2>Supported Ops</h2>
<p>Hexagon only supports ops that have inputs/outputs of &lt;= 4 dimensions.
The following operations have been implemented, with a few constraints that
are verified in <code>IsNodeSupportedByHexagon</code>:</p>
<ul>
<li>Add (Support relu activations)</li>
<li>ArgMax</li>
<li>ArgMin</li>
<li>AveragePool2D:</li>
<li>Constraints:<ul>
<li>No Activation</li>
</ul>
</li>
<li>Concat</li>
<li>Conv2D:</li>
<li>Constraints:<ul>
<li>stride width/height &lt;= 3</li>
</ul>
</li>
<li>DepthToSpace</li>
<li>DepthwiseConv2D:</li>
<li>Constraints:<ul>
<li>Filter height &gt;= 2</li>
<li>depth_multiplier == 1</li>
<li>dilation only supported when stride == 1</li>
<li>Otherwise, stride height/width &lt;= 3</li>
</ul>
</li>
<li>FullyConnected</li>
<li>Hardswish</li>
<li>L2Normalization (without any activation)</li>
<li>Logistic (aka Sigmoid)</li>
<li>Maximum</li>
<li>MaxPool2D (without any activation) (b/129276536)</li>
<li>Mean</li>
<li>Minimum</li>
<li>MirrorPad</li>
<li>Mul (Support relu activations)</li>
<li>Neg</li>
<li>Pack</li>
<li>Pad: Only supports 0 padding (b/139277813)</li>
<li>Quantize (8-bit inputs &amp; outputs only)</li>
<li>Relu</li>
<li>Relu6</li>
<li>Reshape</li>
<li>Resize Bilinear:</li>
<li>Constraints:<ul>
<li>Requested size &lt;= 65 (b/143105433)</li>
</ul>
</li>
<li>Resize Nearest Neighbor</li>
<li>Rsqrt</li>
<li>Slice</li>
<li>SoftMax</li>
<li>SpaceToDepth</li>
<li>Split</li>
<li>SquaredDifference</li>
<li>Strided Slice</li>
<li>Sub (Support relu activations)</li>
<li>Tanh</li>
<li>Transpose</li>
<li>TransposeConv2D:</li>
<li>Constraints:<ul>
<li>stride height/width &lt;= 3</li>
<li>dilation height/width == 1</li>
</ul>
</li>
</ul>