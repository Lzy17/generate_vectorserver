<h1>Jenkins</h1>
<p>The scripts in this directory are the entrypoint for testing ONNX exporter.</p>
<p>The environment variable <code>BUILD_ENVIRONMENT</code> is expected to be set to
the build environment you intend to test. It is a hint for the build
and test scripts to configure Caffe2 a certain way and include/exclude
tests. Docker images, they equal the name of the image itself. For
example: <code>py2-cuda9.0-cudnn7-ubuntu16.04</code>. The Docker images that are
built on Jenkins and are used in triggered builds already have this
environment variable set in their manifest. Also see
<code>./docker/jenkins/*/Dockerfile</code> and search for <code>BUILD_ENVIRONMENT</code>.</p>
<p>Our Jenkins installation is located at https://ci.pytorch.org/jenkins/.</p>