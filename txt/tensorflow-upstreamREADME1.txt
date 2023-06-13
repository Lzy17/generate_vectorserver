<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_horizontal.png">
</div>

<p><a href="https://badge.fury.io/py/tensorflow"><img alt="Python" src="https://img.shields.io/pypi/pyversions/tensorflow.svg" /></a>
<a href="https://badge.fury.io/py/tensorflow"><img alt="PyPI" src="https://badge.fury.io/py/tensorflow.svg" /></a>
<a href="https://doi.org/10.5281/zenodo.4724125"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.4724125.svg" /></a>
<a href="https://bestpractices.coreinfrastructure.org/projects/1486"><img alt="CII Best Practices" src="https://bestpractices.coreinfrastructure.org/projects/1486/badge" /></a>
<a href="https://api.securityscorecards.dev/projects/github.com/tensorflow/tensorflow"><img alt="OpenSSF Scorecard" src="https://api.securityscorecards.dev/projects/github.com/tensorflow/tensorflow/badge" /></a>
<a href="https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&amp;can=1&amp;q=proj:tensorflow"><img alt="Fuzzing Status" src="https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow.svg" /></a>
<a href="https://bugs.chromium.org/p/oss-fuzz/issues/list?sort=-opened&amp;can=1&amp;q=proj:tensorflow-py"><img alt="Fuzzing Status" src="https://oss-fuzz-build-logs.storage.googleapis.com/badges/tensorflow-py.svg" /></a>
<a href="https://ossrank.com/p/44"><img alt="OSSRank" src="https://shields.io/endpoint?url=https://ossrank.com/shield/44" /></a>
<a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg" /></a>
<a href="https://tensorflow.github.io/build#TF%20Official%20Continuous"><img alt="TF Official Continuous" src="https://tensorflow.github.io/build/TF%20Official%20Continuous.svg" /></a>
<a href="https://tensorflow.github.io/build#TF%20Official%20Nightly"><img alt="TF Official Nightly" src="https://tensorflow.github.io/build/TF%20Official%20Nightly.svg" /></a></p>
<p><strong><code>Documentation</code></strong> |
------------------- |
<a href="https://www.tensorflow.org/api_docs/"><img alt="Documentation" src="https://img.shields.io/badge/api-reference-blue.svg" /></a> |</p>
<p><a href="https://www.tensorflow.org/">TensorFlow</a> is an end-to-end open source platform
for machine learning. It has a comprehensive, flexible ecosystem of
<a href="https://www.tensorflow.org/resources/tools">tools</a>,
<a href="https://www.tensorflow.org/resources/libraries-extensions">libraries</a>, and
<a href="https://www.tensorflow.org/community">community</a> resources that lets
researchers push the state-of-the-art in ML and developers easily build and
deploy ML-powered applications.</p>
<p>TensorFlow was originally developed by researchers and engineers working on the
Google Brain team within Google's Machine Intelligence Research organization to
conduct machine learning and deep neural networks research. The system is
general enough to be applicable in a wide variety of other domains, as well.</p>
<p>TensorFlow provides stable <a href="https://www.tensorflow.org/api_docs/python">Python</a>
and <a href="https://www.tensorflow.org/api_docs/cc">C++</a> APIs, as well as
non-guaranteed backward compatible API for
<a href="https://www.tensorflow.org/api_docs">other languages</a>.</p>
<p>Keep up-to-date with release announcements and security updates by subscribing
to
<a href="https://groups.google.com/a/tensorflow.org/forum/#!forum/announce">announce@tensorflow.org</a>.
See all the <a href="https://www.tensorflow.org/community/forums">mailing lists</a>.</p>
<h2>Tensorflow ROCm port</h2>
<p>Please follow the instructions <a href="https://github.com/RadeonOpenCompute/ROCm-docker/blob/master/quick-start.md">here</a> to set up your ROCm stack.
A docker container: <strong>rocm/tensorflow:latest(https://hub.docker.com/r/rocm/tensorflow/)</strong> is readily available to be used:
```
alias drun='sudo docker run \
      -it \
      --network=host \
      --device=/dev/kfd \
      --device=/dev/dri \
      --ipc=host \
      --shm-size 16G \
      --group-add video \
      --cap-add=SYS_PTRACE \
      --security-opt seccomp=unconfined \
      -v $HOME/dockerx:/dockerx'</p>
<p>drun rocm/tensorflow:latest
```</p>
<p>We maintain <code>tensorflow-rocm</code> whl packages on PyPI <a href="https://pypi.org/project/tensorflow-rocm">here</a>, to install tensorflow-rocm package using pip:
```</p>
<h1>Install some ROCm dependencies</h1>
<p>sudo apt install rocm-libs rccl</p>
<h1>Pip3 install the whl package from PyPI</h1>
<p>pip3 install --user tensorflow-rocm --upgrade
```
For details on Tensorflow ROCm port, please take a look at the <a href="README.ROCm.md">ROCm-specific README file</a>.</p>
<h2>Install</h2>
<p>See the <a href="https://www.tensorflow.org/install">TensorFlow install guide</a> for the
<a href="https://www.tensorflow.org/install/pip">pip package</a>, to
<a href="https://www.tensorflow.org/install/gpu">enable GPU support</a>, use a
<a href="https://www.tensorflow.org/install/docker">Docker container</a>, and
<a href="https://www.tensorflow.org/install/source">build from source</a>.</p>
<p>To install the current release, which includes support for
<a href="https://www.tensorflow.org/install/gpu">CUDA-enabled GPU cards</a> <em>(Ubuntu and
Windows)</em>:</p>
<p><code>$ pip install tensorflow</code></p>
<p>Other devices (DirectX and MacOS-metal) are supported using
<a href="https://www.tensorflow.org/install/gpu_plugins#available_devices">Device plugins</a>.</p>
<p>A smaller CPU-only package is also available:</p>
<p><code>$ pip install tensorflow-cpu</code></p>
<p>To update TensorFlow to the latest version, add <code>--upgrade</code> flag to the above
commands.</p>
<p><em>Nightly binaries are available for testing using the
<a href="https://pypi.python.org/pypi/tf-nightly">tf-nightly</a> and
<a href="https://pypi.python.org/pypi/tf-nightly-cpu">tf-nightly-cpu</a> packages on PyPi.</em></p>
<h4><em>Try your first TensorFlow program</em></h4>
<p><code>shell
$ python</code></p>
<p>```python</p>
<blockquote>
<blockquote>
<blockquote>
<p>import tensorflow as tf
tf.add(1, 2).numpy()
3
hello = tf.constant('Hello, TensorFlow!')
hello.numpy()
b'Hello, TensorFlow!'
```</p>
</blockquote>
</blockquote>
</blockquote>
<p>For more examples, see the
<a href="https://www.tensorflow.org/tutorials/">TensorFlow tutorials</a>.</p>
<h2>Contribution guidelines</h2>
<p><strong>If you want to contribute to TensorFlow, be sure to review the
<a href="CONTRIBUTING.md">contribution guidelines</a>. This project adheres to TensorFlow's
<a href="CODE_OF_CONDUCT.md">code of conduct</a>. By participating, you are expected to
uphold this code.</strong></p>
<p><strong>We use <a href="https://github.com/tensorflow/tensorflow/issues">GitHub issues</a> for
tracking requests and bugs, please see
<a href="https://discuss.tensorflow.org/">TensorFlow Forum</a> for general questions and
discussion, and please direct specific questions to
<a href="https://stackoverflow.com/questions/tagged/tensorflow">Stack Overflow</a>.</strong></p>
<p>The TensorFlow project strives to abide by generally accepted best practices in
open-source software development.</p>
<h2>Patching guidelines</h2>
<p>Follow these steps to patch a specific version of TensorFlow, for example, to
apply fixes to bugs or security vulnerabilities:</p>
<ul>
<li>Clone the TensorFlow repo and switch to the corresponding branch for your
    desired TensorFlow version, for example, branch <code>r2.8</code> for version 2.8.</li>
<li>Apply (that is, cherry pick) the desired changes and resolve any code
    conflicts.</li>
<li>Run TensorFlow tests and ensure they pass.</li>
<li><a href="https://www.tensorflow.org/install/source">Build</a> the TensorFlow pip
    package from source.</li>
</ul>
<h2>Continuous build status</h2>
<p>You can find more community-supported platforms and configurations in the
<a href="https://github.com/tensorflow/build#community-supported-tensorflow-builds">TensorFlow SIG Build community builds table</a>.</p>
<h3>Official Builds</h3>
<p>Build Type                    | Status                                                                                                                                                                           | Artifacts
----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------
<strong>Linux CPU</strong>                 | <a href="https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.html"><img alt="Status" src="https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.svg" /></a>           | <a href="https://pypi.org/project/tf-nightly/">PyPI</a>
<strong>Linux GPU</strong>                 | <a href="https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.html"><img alt="Status" src="https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.svg" /></a> | <a href="https://pypi.org/project/tf-nightly-gpu/">PyPI</a>
<strong>Linux XLA</strong>                 | <a href="https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.html"><img alt="Status" src="https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.svg" /></a>         | TBA
<strong>macOS</strong>                     | <a href="https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.html"><img alt="Status" src="https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.svg" /></a>     | <a href="https://pypi.org/project/tf-nightly/">PyPI</a>
<strong>Windows CPU</strong>               | <a href="https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.html"><img alt="Status" src="https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.svg" /></a>       | <a href="https://pypi.org/project/tf-nightly/">PyPI</a>
<strong>Windows GPU</strong>               | <a href="https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.html"><img alt="Status" src="https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.svg" /></a>       | <a href="https://pypi.org/project/tf-nightly-gpu/">PyPI</a>
<strong>Android</strong>                   | <a href="https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.html"><img alt="Status" src="https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.svg" /></a>               | <a href="https://bintray.com/google/tensorflow/tensorflow/_latestVersion">Download</a>
<strong>Raspberry Pi 0 and 1</strong>      | <a href="https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.html"><img alt="Status" src="https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.svg" /></a>           | <a href="https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv6l.whl">Py3</a>
<strong>Raspberry Pi 2 and 3</strong>      | <a href="https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.html"><img alt="Status" src="https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.svg" /></a>           | <a href="https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv7l.whl">Py3</a>
<strong>Libtensorflow MacOS CPU</strong>   | Status Temporarily Unavailable                                                                                                                                                   | <a href="https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/macos/latest/macos_cpu_libtensorflow_binaries.tar.gz">Nightly Binary</a> <a href="https://storage.googleapis.com/tensorflow/">Official GCS</a>
<strong>Libtensorflow Linux CPU</strong>   | Status Temporarily Unavailable                                                                                                                                                   | <a href="https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/ubuntu_16/latest/cpu/ubuntu_cpu_libtensorflow_binaries.tar.gz">Nightly Binary</a> <a href="https://storage.googleapis.com/tensorflow/">Official GCS</a>
<strong>Libtensorflow Linux GPU</strong>   | Status Temporarily Unavailable                                                                                                                                                   | <a href="https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/ubuntu_16/latest/gpu/ubuntu_gpu_libtensorflow_binaries.tar.gz">Nightly Binary</a> <a href="https://storage.googleapis.com/tensorflow/">Official GCS</a>
<strong>Libtensorflow Windows CPU</strong> | Status Temporarily Unavailable                                                                                                                                                   | <a href="https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/windows/latest/cpu/windows_cpu_libtensorflow_binaries.tar.gz">Nightly Binary</a> <a href="https://storage.googleapis.com/tensorflow/">Official GCS</a>
<strong>Libtensorflow Windows GPU</strong> | Status Temporarily Unavailable                                                                                                                                                   | <a href="https://storage.googleapis.com/libtensorflow-nightly/prod/tensorflow/release/windows/latest/gpu/windows_gpu_libtensorflow_binaries.tar.gz">Nightly Binary</a> <a href="https://storage.googleapis.com/tensorflow/">Official GCS</a></p>
<h2>Resources</h2>
<ul>
<li><a href="https://www.tensorflow.org">TensorFlow.org</a></li>
<li><a href="https://www.tensorflow.org/tutorials/">TensorFlow Tutorials</a></li>
<li><a href="https://github.com/tensorflow/models/tree/master/official">TensorFlow Official Models</a></li>
<li><a href="https://github.com/tensorflow/examples">TensorFlow Examples</a></li>
<li><a href="https://codelabs.developers.google.com/?cat=TensorFlow">TensorFlow Codelabs</a></li>
<li><a href="https://blog.tensorflow.org">TensorFlow Blog</a></li>
<li><a href="https://www.tensorflow.org/resources/learn-ml">Learn ML with TensorFlow</a></li>
<li><a href="https://twitter.com/tensorflow">TensorFlow Twitter</a></li>
<li><a href="https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ">TensorFlow YouTube</a></li>
<li><a href="https://www.tensorflow.org/model_optimization/guide/roadmap">TensorFlow model optimization roadmap</a></li>
<li><a href="https://www.tensorflow.org/about/bib">TensorFlow White Papers</a></li>
<li><a href="https://github.com/tensorflow/tensorboard">TensorBoard Visualization Toolkit</a></li>
<li><a href="https://cs.opensource.google/tensorflow/tensorflow">TensorFlow Code Search</a></li>
</ul>
<p>Learn more about the
<a href="https://www.tensorflow.org/community">TensorFlow community</a> and how to
<a href="https://www.tensorflow.org/community/contribute">contribute</a>.</p>
<h2>Courses</h2>
<ul>
<li><a href="https://www.edx.org/course/deep-learning-with-tensorflow">Deep Learning with Tensorflow from Edx</a></li>
<li><a href="https://www.coursera.org/specializations/tensorflow-in-practice">DeepLearning.AI TensorFlow Developer Professional Certificate from Coursera</a></li>
<li><a href="https://www.coursera.org/specializations/tensorflow-data-and-deployment">TensorFlow: Data and Deployment from Coursera</a></li>
<li><a href="https://www.coursera.org/learn/getting-started-with-tensor-flow2">Getting Started with TensorFlow 2 from Coursera</a></li>
<li><a href="https://www.coursera.org/specializations/tensorflow-advanced-techniques">TensorFlow: Advanced Techniques from Coursera</a></li>
<li><a href="https://www.coursera.org/specializations/tensorflow2-deeplearning">TensorFlow 2 for Deep Learning Specialization from Coursera</a></li>
<li><a href="https://www.coursera.org/learn/introduction-tensorflow">Intro to TensorFlow for A.I, M.L, and D.L from Coursera</a></li>
<li><a href="https://www.coursera.org/specializations/machine-learning-tensorflow-gcp">Machine Learning with TensorFlow on GCP from Coursera</a></li>
<li><a href="https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187">Intro to TensorFlow for Deep Learning from Udacity</a></li>
<li><a href="https://www.udacity.com/course/intro-to-tensorflow-lite--ud190">Introduction to TensorFlow Lite from Udacity</a></li>
</ul>
<h2>License</h2>
<p><a href="LICENSE">Apache License 2.0</a></p>