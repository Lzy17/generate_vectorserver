<h1>Docker images for Jenkins</h1>
<p>This directory contains everything needed to build the Docker images
that are used in our CI</p>
<p>The Dockerfiles located in subdirectories are parameterized to
conditionally run build stages depending on build arguments passed to
<code>docker build</code>. This lets us use only a few Dockerfiles for many
images. The different configurations are identified by a freeform
string that we call a <em>build environment</em>. This string is persisted in
each image as the <code>BUILD_ENVIRONMENT</code> environment variable.</p>
<p>See <code>build.sh</code> for valid build environments (it's the giant switch).</p>
<p>Docker builds are now defined with <code>.circleci/cimodel/data/simple/docker_definitions.py</code></p>
<h2>Contents</h2>
<ul>
<li><code>build.sh</code> -- dispatch script to launch all builds</li>
<li><code>common</code> -- scripts used to execute individual Docker build stages</li>
<li><code>ubuntu-cuda</code> -- Dockerfile for Ubuntu image with CUDA support for nvidia-docker</li>
</ul>
<h2>Usage</h2>
<p>```bash</p>
<h1>Build a specific image</h1>
<p>./build.sh pytorch-linux-bionic-py3.8-gcc9 -t myimage:latest</p>
<h1>Set flags (see build.sh) and build image</h1>
<p>sudo bash -c 'PROTOBUF=1 ./build.sh pytorch-linux-bionic-py3.8-gcc9 -t myimage:latest
```</p>