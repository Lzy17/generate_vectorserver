<h1>SavedModel importer FileCheck tests.</h1>
<h2>Debugging tests</h2>
<p>While debugging tests, the following commands are handy.</p>
<p>Run FileCheck test:</p>
<p><code>bazel run :foo.py.test</code></p>
<p>Run just the Python file and look at the output:</p>
<p><code>bazel run :foo</code></p>
<p>Generate saved model to inspect proto:</p>
<p>```
bazel run :foo -- --save_model_path=/tmp/my.saved_model</p>
<h1>Inspect /tmp/my.saved_model/saved_model.pb</h1>
<p>```</p>
<h2>Rationale for Python-based tests</h2>
<p>For a SavedModel importer, the natural place to start is to feed in the
SavedModel format directly and test the output MLIR. We don't do that though.</p>
<p>The SavedModel format is a directory structure which contains a SavedModel proto
and some other stuff (mostly binary files of some sort) in it. That makes it not
suitable for use as a test input, since it is not human-readable. Even just the
text proto for the SavedModel proto is difficult to use as a test input, since a
small piece of Python code (e.g. just a tf.Add) generates thousands of lines of
text proto.</p>
<p>That points to a solution though: write our tests starting from the Python API's
that generate the SavedModel. That leads to very compact test inputs.</p>
<p>As the SavedModel work progresses, it's likely to be of interest to find a
shortcut between the Python <code>tf.Module</code> and the SavedModel MLIR representation
that doesn't involve serializing a SavedModel to disk and reading it back.</p>
<h2>Potential improvements</h2>
<p>The test iteration cycle for these tests is very long (usually over a minute).
We need to find a way to improve this in the future.</p>