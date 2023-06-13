<p>This directory contains scripts for our continuous integration.</p>
<p>One important thing to keep in mind when reading the scripts here is
that they are all based off of Docker images, which we build for each of
the various system configurations we want to run on Jenkins.  This means
it is very easy to run these tests yourself:</p>
<ol>
<li>
<p>Figure out what Docker image you want.  The general template for our
   images look like:
   <code>registry.pytorch.org/pytorch/pytorch-$BUILD_ENVIRONMENT:$DOCKER_VERSION</code>,
   where <code>$BUILD_ENVIRONMENT</code> is one of the build environments
   enumerated in
   <a href="https://github.com/pytorch/pytorch/blob/master/.ci/docker/build.sh">pytorch-dockerfiles</a>. The dockerfile used by jenkins can be found under the <code>.ci</code> <a href="https://github.com/pytorch/pytorch/blob/master/.ci/docker">directory</a></p>
</li>
<li>
<p>Run <code>docker run -it -u jenkins $DOCKER_IMAGE</code>, clone PyTorch and
   run one of the scripts in this directory.</p>
</li>
</ol>
<p>The Docker images are designed so that any "reasonable" build commands
will work; if you look in <a href="build.sh">build.sh</a> you will see that it is a
very simple script.  This is intentional.  Idiomatic build instructions
should work inside all of our Docker images.  You can tweak the commands
however you need (e.g., in case you want to rebuild with DEBUG, or rerun
the build with higher verbosity, etc.).</p>
<p>We have to do some work to make this so.  Here is a summary of the
mechanisms we use:</p>
<ul>
<li>
<p>We install binaries to directories like <code>/usr/local/bin</code> which
  are automatically part of your PATH.</p>
</li>
<li>
<p>We add entries to the PATH using Docker ENV variables (so
  they apply when you enter Docker) and <code>/etc/environment</code> (so they
  continue to apply even if you sudo), instead of modifying
  <code>PATH</code> in our build scripts.</p>
</li>
<li>
<p>We use <code>/etc/ld.so.conf.d</code> to register directories containing
  shared libraries, instead of modifying <code>LD_LIBRARY_PATH</code> in our
  build scripts.</p>
</li>
<li>
<p>We reroute well known paths like <code>/usr/bin/gcc</code> to alternate
  implementations with <code>update-alternatives</code>, instead of setting
  <code>CC</code> and <code>CXX</code> in our implementations.</p>
</li>
</ul>