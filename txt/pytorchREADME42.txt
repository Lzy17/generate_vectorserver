<h1>Caffe2 implementation of Open Neural Network Exchange (ONNX)</h1>
<h1>Usage</h1>
<ul>
<li><a href="https://github.com/onnx/tutorials/blob/master/tutorials/OnnxCaffe2Import.ipynb">ONNX to Caffe2</a></li>
<li><a href="https://github.com/onnx/tutorials/blob/master/tutorials/Caffe2OnnxExport.ipynb">Caffe2 to ONNX</a></li>
<li><a href="https://github.com/onnx/tutorials">other end-to-end tutorials</a></li>
</ul>
<h1>Installation</h1>
<p>onnx-caffe2 is installed as a part of Caffe2.
Please follow the <a href="https://caffe2.ai/docs/getting-started.html">instructions</a> to install Caffe2.</p>
<h1>Folder Structure</h1>
<ul>
<li>./: the main folder that all code lies under</li>
<li>frontend.py: translate from caffe2 model to onnx model</li>
<li>backend.py: execution engine that runs onnx on caffe2</li>
<li>tests/: test files</li>
</ul>
<h1>Testing</h1>
<p>onnx-caffe2 uses <a href="https://docs.pytest.org">pytest</a> as test driver. In order to run tests, first you need to install pytest:</p>
<p><code>pip install pytest-cov</code></p>
<p>After installing pytest, do</p>
<p><code>pytest</code></p>
<p>to run tests.</p>
<p>Testing coverage issues/status: https://github.com/caffe2/caffe2/blob/master/caffe2/python/onnx/ONNXOpCoverage.md</p>
<h1>Development</h1>
<p>During development it's convenient to install caffe2 in development mode:</p>
<p><code>cd /path/to/caffe2
pip install -e caffe2/</code></p>
<h1>License</h1>
<p><a href="LICENSE">MIT License</a></p>