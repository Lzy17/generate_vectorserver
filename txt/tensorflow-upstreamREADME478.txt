<p>Bazel rules to package the TensorFlow APIs in languages other than Python into
archives.</p>
<h2>C library</h2>
<p>The TensorFlow <a href="https://www.tensorflow.org/code/tensorflow/c/c_api.h">C
API</a>
is typically a requirement of TensorFlow APIs in other languages such as
<a href="https://www.tensorflow.org/code/tensorflow/go">Go</a>
and <a href="https://github.com/tensorflow/rust">Rust</a>.</p>
<p>The following commands:</p>
<p><code>sh
bazel test --config opt //tensorflow/tools/lib_package:libtensorflow_test
bazel build --config opt //tensorflow/tools/lib_package:libtensorflow</code></p>
<p>test and produce the archive at
<code>bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz</code>, which can be
distributed and installed using something like:</p>
<p><code>sh
tar -C /usr/local -xzf libtensorflow.tar.gz</code></p>
<h2>Java library</h2>
<p>The TensorFlow <a href="https://www.tensorflow.org/code/tensorflow/java/README.md">Java
API</a>
consists of a native library (<code>libtensorflow_jni.so</code>) and a Java archive (JAR).
The following commands:</p>
<p><code>sh
bazel test --config opt //tensorflow/tools/lib_package:libtensorflow_test
bazel build --config opt \
  //tensorflow/tools/lib_package:libtensorflow_jni.tar.gz \
  //tensorflow/java:libtensorflow.jar \
  //tensorflow/java:libtensorflow-src.jar</code></p>
<p>test and produce the following:</p>
<ul>
<li>The native library (<code>libtensorflow_jni.so</code>) packaged in an archive at:
    <code>bazel-bin/tensorflow/tools/lib_package/libtensorflow_jni.tar.gz</code></li>
<li>The Java archive at:
    <code>bazel-bin/tensorflow/java/libtensorflow.jar</code></li>
<li>The Java archive for Java sources at:
    <code>bazel-bin/tensorflow/java/libtensorflow-src.jar</code></li>
</ul>
<h2>Release</h2>
<p>Scripts to build these archives for TensorFlow releases are in
<a href="https://www.tensorflow.org/code/tensorflow/tools/ci_build/linux">tensorflow/tools/ci_build/linux</a>
and
<a href="https://www.tensorflow.org/code/tensorflow/tools/ci_build/osx">tensorflow/tools/ci_build/osx</a></p>