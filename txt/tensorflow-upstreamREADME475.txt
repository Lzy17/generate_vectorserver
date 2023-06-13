<h1>TF SIG Build Dockerfiles</h1>
<p>Standard Dockerfiles for TensorFlow builds, used internally at Google.</p>
<p>Maintainer: @angerson (TensorFlow OSS DevInfra; SIG Build)</p>
<hr />
<p>These docker containers are for building and testing TensorFlow in CI
environments (and for users replicating those CI builds). They are openly
developed in TF SIG Build, verified by Google developers, and published to
tensorflow/build on <a href="https://hub.docker.com/r/tensorflow/build/">Docker Hub</a>.
The TensorFlow OSS DevInfra team uses these containers for most of our
Linux-based CI, including <code>tf-nightly</code> tests and Pip packages and TF release
packages for TensorFlow 2.9 onwards.</p>
<h2>Tags</h2>
<p>These Dockerfiles are built and deployed to <a href="https://hub.docker.com/r/tensorflow/build/">Docker
Hub</a> via <a href="https://github.com/tensorflow/tensorflow/blob/master/.github/workflows/sigbuild-docker.yml">Github
Actions</a>.</p>
<p>The tags are defined as such:</p>
<ul>
<li>The <code>latest</code> tags are kept up-to-date to build TensorFlow's <code>master</code> branch.</li>
<li>The <code>version number</code> tags target the corresponding TensorFlow version. We
  continuously build the <code>current-tensorflow-version + 1</code> tag, so when a new
  TensorFlow branch is cut, that Dockerfile is frozen to support that branch.</li>
<li>We support the same Python versions that TensorFlow does.</li>
</ul>
<h2>Updating the Containers</h2>
<p>For simple changes, you can adjust the source files and then make a PR. Send it
to @angerson for review. We have presubmits that will make sure your change
still builds a container. After approval and submission, our GitHub Actions
workflow deploys the containers to Docker Hub.</p>
<ul>
<li>To update Python packages, look at <code>devel.requirements.txt</code></li>
<li>To update system packages, look at <code>devel.packages.txt</code></li>
<li>To update the way <code>bazel build</code> works, look at <code>devel.usertools/*.bazelrc</code>.</li>
</ul>
<p>To rebuild the containers locally after making changes, use this command from
this directory:</p>
<p>For CUDA
<code>bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg PYTHON_VERSION=python3.9 --target=devel -t my-tf-devel .</code>
For ROCM
<code>DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm \
  --build-arg ROCM_VERSION=5.5.0 --build-arg PYTHON_VERSION=3.9 -t my-tf-devel .</code>
It will take a long time to build devtoolset and install packages. After
it's done, you can use the commands below to test your changes. Just replace
<code>tensorflow/build:latest-python3.9</code> with <code>my-tf-devel</code> to use your image
instead.</p>
<h3>Automatic GCR.io Builds for Presubmits</h3>
<p>TensorFlow team members (i.e. Google employees) can apply a <code>Build and deploy
to gcr.io for staging</code> tag to their PRs to the Dockerfiles, as long as the PR
is being developed on a branch of this repository, not a fork. Unfortunately
this is not available for non-Googler contributors for security reasons.</p>
<h2>Run the TensorFlow Team's Nightly Test Suites with Docker</h2>
<p>The TensorFlow DevInfra team runs a daily test suite that builds <code>tf-nightly</code>
and runs a <code>bazel test</code> suite on both the Pip package (the "pip" tests) and
on the source code itself (the "nonpip" tests). These test scripts are often
referred to as "The Nightly Tests" and can be a common reason for a TF PR to be
reverted. The build scripts aren't visible to external users, but they use
the configuration files which are included in these containers. Our test suites,
which include the build of <code>tf-nightly</code>, are easy to replicate with these
containers, and here is how you can do it.</p>
<p>Presubmits are not using these containers... yet.</p>
<p>Here are some important notes to keep in mind:</p>
<ul>
<li>
<p>The Ubuntu CI jobs that build the <code>tf-nightly</code> package build at the GitHub
  <code>nightly</code> tag. You can see the specific commit of a <code>tf-nightly</code> package on
  pypi.org in <code>tf.version.GIT_VERSION</code>, which will look something like
  <code>v1.12.1-67282-g251085598b7</code>. The final section, <code>g251085598b7</code>, is a short
  git hash.</p>
</li>
<li>
<p>If you interrupt a <code>docker exec</code> command with <code>ctrl-c</code>, you will get your
  shell back but the command will continue to run. You cannot reattach to it,
  but you can kill it with <code>docker kill tf</code> (or <code>docker kill the-container-name</code>).
  This will destroy your container but will not harm your work since it's
  mounted.  If you have any suggestions for handling this better, let us know.</p>
</li>
</ul>
<p>Now let's build <code>tf-nightly</code>.</p>
<ol>
<li>
<p>Set up your directories:</p>
<ul>
<li>A directory with the TensorFlow source code, e.g. <code>/tmp/tensorflow</code></li>
<li>A directory for TensorFlow packages built in the container, e.g. <code>/tmp/packages</code></li>
<li>A directory for your local bazel cache (can be empty), e.g. <code>/tmp/bazelcache</code></li>
</ul>
</li>
<li>
<p>Choose the Docker container to use from <a href="https://hub.docker.com/r/tensorflow/build/tags">Docker
   Hub</a>. The options for the
   <code>master</code> branch are:</p>
</li>
</ol>
<p>For CUDA</p>
<pre><code>- `tensorflow/build:latest-python3.11`
- `tensorflow/build:latest-python3.10`
- `tensorflow/build:latest-python3.9`
- `tensorflow/build:latest-python3.8`
</code></pre>
<p>For ROCM</p>
<pre><code>- `rocm/tensorflow-build:latest-python3.10`
- `rocm/tensorflow-build:latest-python3.9`
- `rocm/tensorflow-build:latest-python3.8`
- `rocm/tensorflow-build:latest-python3.7`

For this example we'll use `tensorflow/build:latest-python3.9`.
</code></pre>
<ol>
<li>Pull the container you decided to use.</li>
</ol>
<p>For CUDA</p>
<pre><code>```bash
docker pull tensorflow/build:latest-python3.9
```
</code></pre>
<p>For ROCM</p>
<pre><code>```bash
docker pull rocm/tensorflow-build:latest-python3.9
```
</code></pre>
<ol>
<li>
<p>Start a backgrounded Docker container with the three folders mounted.</p>
<ul>
<li>Mount the TensorFlow source code to <code>/tf/tensorflow</code>.</li>
<li>Mount the directory for built packages to <code>/tf/pkg</code>.</li>
<li>Mount the bazel cache to <code>/tf/cache</code>. You don't need <code>/tf/cache</code> if
  you're going to use the remote cache.</li>
</ul>
<p>Here are the arguments we're using:</p>
<ul>
<li><code>--name tf</code>: Names the container <code>tf</code> so we can refer to it later.</li>
<li><code>-w /tf/tensorflow</code>: All commands run in the <code>/tf/tensorflow</code> directory,
  where the TF source code is.</li>
<li><code>-it</code>: Makes the container interactive for running commands</li>
<li><code>-d</code>: Makes the container start in the background, so we can send
  commands to it instead of running commands from inside.</li>
</ul>
<p>And <code>-v</code> is for mounting directories into the container.</p>
<p>For CUDA
<code>bash
docker run --name tf -w /tf/tensorflow -it -d \
  -v "/tmp/packages:/tf/pkg" \
  -v "/tmp/tensorflow:/tf/tensorflow" \
  -v "/tmp/bazelcache:/tf/cache" \
  tensorflow/build:latest-python3.9 \
  bash</code></p>
<p>For ROCM
<code>docker run --name tf -w /tf/tensorflow -it -d --network=host \
--device=/dev/kfd --device=/dev/dri \
--ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v "/tmp/packages:/tf/pkg" \
-v "/tmp/tensorflow:/tf/tensorflow" \
-v "/tmp/bazelcache:/tf/cache" \
rocm/tensorflow-build:latest-python3.9 \
bash</code></p>
<p>Note: if you wish to use your own Google Cloud Platform credentials for
e.g. RBE, you may also wish to set <code>-v
$HOME/.config/gcloud:/root/.config/gcloud</code> to make your credentials
available to bazel. You don't need to do this unless you know what you're
doing.</p>
</li>
</ol>
<p>Now you can continue on to any of:</p>
<ul>
<li>Build <code>tf-nightly</code> and then (optionally) run a test suite on the pip package
  (the "pip" suite)</li>
<li>Run a test suite on the TF code directly (the "nonpip" suite)</li>
<li>Build the libtensorflow packages (the "libtensorflow" suite)</li>
<li>Run a code-correctness check (the "code_check" suite)</li>
</ul>
<h3>Build <code>tf-nightly</code> and run Pip tests</h3>
<ol>
<li>
<p>Apply the <code>update_version.py</code> script that changes the TensorFlow version to
   <code>X.Y.Z.devYYYYMMDD</code>. This is used for <code>tf-nightly</code> on PyPI and is technically
   optional.</p>
<p><code>bash
docker exec tf python3 tensorflow/tools/ci_build/update_version.py --nightly</code></p>
</li>
<li>
<p>Build TensorFlow by following the instructions under one of the collapsed
   sections below. You can build both CPU and GPU packages without a GPU. TF
   DevInfra's remote cache is better for building TF only once, but if you
   build over and over, it will probably be better in the long run to use a
   local cache. We're not sure about which is best for most users, so let us
   know on <a href="https://gitter.im/tensorflow/sig-build">Gitter</a>.</p>
</li>
</ol>
<p>This step will take a long time, since you're building TensorFlow. GPU takes
   much longer to build. Choose one and click on the arrow to expand the
   commands:</p>
<pre><code>&lt;details&gt;&lt;summary&gt;TF Nightly CPU - Remote Cache&lt;/summary&gt;

Build the sources with Bazel:

```
docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
build --config=sigbuild_remote_cache \
tensorflow/tools/pip_package:build_pip_package
```

And then construct the pip package:

```
docker exec tf \
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
  /tf/pkg \
  --cpu \
  --nightly_flag
```

&lt;/details&gt;

&lt;details&gt;&lt;summary&gt;TF Nightly GPU - Remote Cache&lt;/summary&gt;

Build the sources with Bazel:

```
docker exec tf bazel --bazelrc=/usertools/gpu.bazelrc \
build --config=sigbuild_remote_cache \
tensorflow/tools/pip_package:build_pip_package
```

And then construct the pip package:

```
docker exec tf \
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
  /tf/pkg \
  --nightly_flag
```

&lt;/details&gt;

&lt;details&gt;&lt;summary&gt;TF Nightly CPU - Local Cache&lt;/summary&gt;

Make sure you have a directory mounted to the container in `/tf/cache`!

Build the sources with Bazel:

```
docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
build --config=sigbuild_local_cache \
tensorflow/tools/pip_package:build_pip_package
```

And then construct the pip package:

```
docker exec tf \
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
  /tf/pkg \
  --cpu \
  --nightly_flag
```

&lt;/details&gt;

&lt;details&gt;&lt;summary&gt;TF Nightly GPU - Local Cache&lt;/summary&gt;

Make sure you have a directory mounted to the container in `/tf/cache`!
For CUDA:

Build the sources with Bazel:

```
docker exec tf \
bazel --bazelrc=/usertools/gpu.bazelrc \
build --config=sigbuild_local_cache \
tensorflow/tools/pip_package:build_pip_package
```

And then construct the pip package:

```
docker exec tf \
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
  /tf/pkg \
  --nightly_flag
```

For ROCM:

Build the sources with Bazel:

```
docker exec tf \
bazel --bazelrc=/usertools/gpu.bazelrc \
build --config=rocm \
tensorflow/tools/pip_package:build_pip_package --verbose_failures

```

And then construct the nightly pip package:

```
docker exec tf \
./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
/tf/pkg \
--rocm \
--nightly_flag
```

Note: if you are creating a release (non-nightly) pip package:
```
docker exec tf \
./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
/tf/pkg \
--rocm \
--project_name tensorflow_rocm
```


&lt;/details&gt;
</code></pre>
<ol>
<li>
<p>Run the helper script that checks for manylinux compliance, renames the
   wheels, and then checks the size of the packages. The auditwheel repair option
   currently doesn't support ROCM and will strip needed symbols, so for ROCM
   we use a separate script that will manually rename the whl.</p>
<p>For CUDA</p>
<p><code>docker exec tf /usertools/rename_and_verify_wheels.sh</code></p>
<p>For ROCM</p>
<p><code>docker exec tf /usertools/rename_and_verify_ROCM_wheels.sh</code></p>
</li>
<li>
<p>Take a look at the new wheel packages you built! They may be owned by <code>root</code>
   because of how Docker volume permissions work.</p>
<p><code>ls -al /tmp/packages</code></p>
</li>
<li>
<p>To continue on to running the Pip tests, create a venv and install the
   testing packages:</p>
<p><code>docker exec tf /usertools/setup_venv_test.sh bazel_pip "/tf/pkg/tf_nightly*.whl"</code></p>
</li>
<li>
<p>And now run the tests depending on your target platform: <code>--config=pip</code>
   includes the same test suite that is run by the DevInfra team every night.
   If you want to run a specific test instead of the whole suite, pass
   <code>--config=pip_venv</code> instead, and then set the target on the command like
   normal.</p>
<p><details><summary>TF Nightly CPU - Remote Cache</summary></p>
<p>Build the sources with Bazel:</p>
<p><code>docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
test --config=sigbuild_remote_cache \
--config=pip</code></p>
</details>
<p><details><summary>TF Nightly GPU - Remote Cache</summary></p>
<p>Build the sources with Bazel:</p>
<p><code>docker exec tf bazel --bazelrc=/usertools/gpu.bazelrc \
test --config=sigbuild_remote_cache \
--config=pip</code></p>
</details>
<p><details><summary>TF Nightly CPU - Local Cache</summary></p>
<p>Make sure you have a directory mounted to the container in <code>/tf/cache</code>!</p>
<p>Build the sources with Bazel:</p>
<p><code>docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
test --config=sigbuild_local_cache \
--config=pip</code></p>
</details>
<p><details><summary>TF Nightly GPU - Local Cache</summary></p>
<p>Make sure you have a directory mounted to the container in <code>/tf/cache</code>!</p>
<p>Build the sources with Bazel:</p>
<p><code>docker exec tf \
bazel --bazelrc=/usertools/gpu.bazelrc \
test --config=sigbuild_local_cache \
--config=pip</code></p>
</details>
</li>
</ol>
<h3>Run Nonpip Tests</h3>
<ol>
<li>
<p>Run the tests depending on your target platform. <code>--config=nonpip</code> includes
   the same test suite that is run by the DevInfra team every night. If you
   want to run a specific test instead of the whole suite, you do not need
   <code>--config=nonpip</code> at all; just set the target on the command line like usual.</p>
<p><details><summary>TF Nightly CPU - Remote Cache</summary></p>
<p>Build the sources with Bazel:</p>
<p><code>docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
test --config=sigbuild_remote_cache \
--config=nonpip</code></p>
</details>
<p><details><summary>TF Nightly GPU - Remote Cache</summary></p>
<p>Build the sources with Bazel:</p>
<p><code>docker exec tf bazel --bazelrc=/usertools/gpu.bazelrc \
test --config=sigbuild_remote_cache \
--config=nonpip</code></p>
</details>
<p><details><summary>TF Nightly CPU - Local Cache</summary></p>
<p>Make sure you have a directory mounted to the container in <code>/tf/cache</code>!</p>
<p>Build the sources with Bazel:</p>
<p><code>docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
test --config=sigbuild_local_cache \
--config=nonpip</code></p>
</details>
<p><details><summary>TF Nightly GPU - Local Cache</summary></p>
<p>Make sure you have a directory mounted to the container in <code>/tf/cache</code>!</p>
<p>Build the sources with Bazel:</p>
<p><code>docker exec tf \
bazel --bazelrc=/usertools/gpu.bazelrc \
test --config=sigbuild_local_cache \
--config=nonpip</code></p>
</details>
</li>
</ol>
<h3>Test, build and package libtensorflow</h3>
<ol>
<li>
<p>Run the tests depending on your target platform.
   <code>--config=libtensorflow_test</code> includes the same test suite that is run by
   the DevInfra team every night. If you want to run a specific test instead of
   the whole suite, just set the target on the command line like usual.</p>
<p><details><summary>TF Nightly CPU - Remote Cache</summary></p>
<p><code>docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
test --config=sigbuild_remote_cache \
--config=libtensorflow_test</code></p>
</details>
<p><details><summary>TF Nightly GPU - Remote Cache</summary></p>
<p><code>docker exec tf bazel --bazelrc=/usertools/gpu.bazelrc \
test --config=sigbuild_remote_cache \
--config=libtensorflow_test</code></p>
</details>
<p><details><summary>TF Nightly CPU - Local Cache</summary></p>
<p>Make sure you have a directory mounted to the container in <code>/tf/cache</code>!</p>
<p><code>docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
test --config=sigbuild_local_cache \
--config=libtensorflow_test</code></p>
</details>
<p><details><summary>TF Nightly GPU - Local Cache</summary></p>
<p>Make sure you have a directory mounted to the container in <code>/tf/cache</code>!</p>
<p><code>docker exec tf \
bazel --bazelrc=/usertools/gpu.bazelrc \
test --config=sigbuild_local_cache \
--config=libtensorflow_test</code></p>
</details>
</li>
<li>
<p>Build the libtensorflow packages.</p>
<p><details><summary>TF Nightly CPU - Remote Cache</summary></p>
<p><code>docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
build --config=sigbuild_remote_cache \
--config=libtensorflow_build</code></p>
</details>
<p><details><summary>TF Nightly GPU - Remote Cache</summary></p>
<p><code>docker exec tf bazel --bazelrc=/usertools/gpu.bazelrc \
build --config=sigbuild_remote_cache \
--config=libtensorflow_build</code></p>
</details>
<p><details><summary>TF Nightly CPU - Local Cache</summary></p>
<p>Make sure you have a directory mounted to the container in <code>/tf/cache</code>!</p>
<p><code>docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
build --config=sigbuild_local_cache \
--config=libtensorflow_build</code></p>
</details>
<p><details><summary>TF Nightly GPU - Local Cache</summary></p>
<p>Make sure you have a directory mounted to the container in <code>/tf/cache</code>!</p>
<p><code>docker exec tf \
bazel --bazelrc=/usertools/gpu.bazelrc \
build --config=sigbuild_local_cache \
--config=libtensorflow_build</code></p>
</details>
</li>
<li>
<p>Run the <code>repack_libtensorflow.sh</code> utility to repack and rename the archives.</p>
<p><details><summary>CPU</summary></p>
<p><code>docker exec tf /usertools/repack_libtensorflow.sh /tf/pkg "-cpu-linux-x86_64"</code></p>
</details>
<p><details><summary>GPU</summary></p>
<p><code>docker exec tf /usertools/repack_libtensorflow.sh /tf/pkg "-gpu-linux-x86_64"</code></p>
</details>
</li>
</ol>
<h3>Run a code check</h3>
<ol>
<li>
<p>Every night the TensorFlow team runs <code>code_check_full</code>, which contains a
   suite of checks that were gradually introduced over TensorFlow's lifetime
   to prevent certain unsable code states. This check has supplanted the old
   "sanity" or "ci_sanity" checks.</p>
<p><code>docker exec tf bats /usertools/code_check_full.bats --timing --formatter junit</code></p>
</li>
</ol>
<h3>Clean Up</h3>
<ol>
<li>
<p>Shut down and remove the container when you are finished.</p>
<p><code>docker stop tf
docker rm tf</code></p>
</li>
</ol>