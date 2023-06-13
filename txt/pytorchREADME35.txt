<h1>Caffe2 &amp; TensorRT integration</h1>
<p><a href="https://ci.pytorch.org/jenkins/job/caffe2-master"><img alt="Jenkins Build Status" src="https://ci.pytorch.org/jenkins/job/caffe2-master/lastCompletedBuild/badge/icon" /></a></p>
<p>This directory contains the code implementing <code>TensorRTOp</code> Caffe2 operator as well as Caffe2 model converter (using <code>ONNX</code> model as an intermediate format).
To enable this functionality in your PyTorch build please set</p>
<p><code>USE_TENSORRT=1 ... python setup.py ...</code></p>
<p>or if you use CMake directly</p>
<p><code>-DUSE_TENSORRT=ON</code></p>
<p>For further information please explore <code>caffe2/python/trt/test_trt.py</code> test showing all possible use cases.</p>
<h2>Questions and Feedback</h2>
<p>Please use GitHub issues (https://github.com/pytorch/pytorch/issues) to ask questions, report bugs, and request new features.</p>