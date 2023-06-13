<h1>AARCH64 toolchain</h1>
<p>Toolchain for performing TensorFlow AARCH64 builds such as used in Github
Actions ARM_CI and ARM_CD.</p>
<p>Maintainer: @elfringham (Linaro LDCG)</p>
<hr />
<p>This repository contains a toolchain for use with the specially constructed
Docker containers that match those created by SIG Build for x86 architecture
builds, but modified for AARCH64 builds.</p>
<p>These Docker containers have been constructed to perform builds of TensorFlow
that are compatible with manylinux2014 requirements but in an environment that
has the C++11 Dual ABI enabled.</p>
<p>The Docker containers are available from
<a href="https://hub.docker.com/r/linaro/tensorflow-arm64-build/tags">Docker Hub</a> The
source Dockerfiles are available from
<a href="https://git.linaro.org/ci/dockerfiles.git/tree/tensorflow-arm64-build">Linaro git</a></p>