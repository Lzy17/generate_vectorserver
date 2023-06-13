<h1>Building TensorFlow Lite Standalone Pip</h1>
<p>Many users would like to deploy TensorFlow lite interpreter and use it from
Python without requiring the rest of TensorFlow.</p>
<h2>Steps</h2>
<p>To build a binary wheel run this script:</p>
<p><code>sh
sudo apt install swig libjpeg-dev zlib1g-dev python3-dev python3-numpy
pip install numpy pybind11
sh tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh</code></p>
<p>That will print out some output and a .whl file. You can then install that</p>
<p><code>sh
pip install --upgrade &lt;wheel&gt;</code></p>
<p>You can also build a wheel inside docker container using make tool. For example
the following command will cross-compile tflite-runtime package for python2.7
and python3.7 (from Debian Buster) on Raspberry Pi:</p>
<p><code>sh
make BASE_IMAGE=debian:buster PYTHON=python TENSORFLOW_TARGET=rpi docker-build
make BASE_IMAGE=debian:buster PYTHON=python3 TENSORFLOW_TARGET=rpi docker-build</code></p>
<p>Another option is to cross-compile for python3.5 (from Debian Stretch) on ARM64
board:</p>
<p><code>sh
make BASE_IMAGE=debian:stretch PYTHON=python3 TENSORFLOW_TARGET=aarch64 docker-build</code></p>
<p>To build for python3.6 (from Ubuntu 18.04) on x86_64 (native to the docker
image) run:</p>
<p><code>sh
make BASE_IMAGE=ubuntu:18.04 PYTHON=python3 TENSORFLOW_TARGET=native docker-build</code></p>
<p>In addition to the wheel there is a way to build Debian package by adding
<code>BUILD_DEB=y</code> to the make command (only for python3):</p>
<p><code>sh
make BASE_IMAGE=debian:buster PYTHON=python3 TENSORFLOW_TARGET=rpi BUILD_DEB=y docker-build</code></p>
<h2>Alternative build with Bazel (experimental)</h2>
<p>There is another build steps to build a binary wheel which uses Bazel instead of
Makefile. You don't need to install additional dependencies.
This approach can leverage TF's <code>ci_build.sh</code> for ARM cross builds.</p>
<h3>Normal build for your workstation</h3>
<p><code>sh
tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh</code></p>
<h3>Optimized build for your workstation</h3>
<p>The output may have a compatibility issue with other machines but it gives the
best performance for your workstation.</p>
<p><code>sh
tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh native</code></p>
<h3>Cross build for armhf Python 3.5</h3>
<p><code>sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON3 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh armhf</code></p>
<h3>Cross build for armhf Python 3.7</h3>
<p><code>sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON37 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh armhf</code></p>
<h3>Cross build for aarch64 Python 3.5</h3>
<p><code>sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON3 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh aarch64</code></p>
<h3>Cross build for aarch64 Python 3.8</h3>
<p><code>sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON38 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh aarch64</code></p>
<h3>Cross build for aarch64 Python 3.9</h3>
<p><code>sh
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON39 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh aarch64</code></p>
<h3>Native build for Windows</h3>
<p><code>sh
bash tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh windows</code></p>
<h2>Enable TF OP support (Flex delegate)</h2>
<p>If you want to use TF ops with Python API, you need to enable flex support.
You can build TFLite interpreter with flex ops support by providing
<code>--define=tflite_pip_with_flex=true</code> to Bazel.</p>
<p>Here are some examples.</p>
<h3>Normal build with Flex for your workstation</h3>
<p><code>sh
CUSTOM_BAZEL_FLAGS=--define=tflite_pip_with_flex=true \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh</code></p>
<h3>Cross build with Flex for armhf Python 3.7</h3>
<p><code>sh
CI_DOCKER_EXTRA_PARAMS="-e CUSTOM_BAZEL_FLAGS=--define=tflite_pip_with_flex=true" \
  tensorflow/tools/ci_build/ci_build.sh PI-PYTHON37 \
  tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh armhf</code></p>
<h2>Usage</h2>
<p>Note, unlike tensorflow this will be installed to a <code>tflite_runtime</code> namespace.
You can then use the Tensorflow Lite interpreter as.</p>
<p><code>python
from tflite_runtime.interpreter import Interpreter
interpreter = Interpreter(model_path="foo.tflite")</code></p>
<p>This currently works to build on Linux machines including Raspberry Pi. In
the future, cross compilation to smaller SOCs like Raspberry Pi from
bigger host will be supported.</p>
<h2>Caveats</h2>
<ul>
<li>You cannot use TensorFlow Select ops, only TensorFlow Lite builtins.</li>
<li>Currently custom ops and delegates cannot be registered.</li>
</ul>