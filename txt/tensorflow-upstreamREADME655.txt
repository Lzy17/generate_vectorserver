<h1>Hexagon Delegate Testing</h1>
<p>This directory contains unit-tests for Op Builders for the hexagon delegate.
To Run the all the tests use the run_tests.sh under directory and pass
the path to the directory containing libhexagon_nn_skel*.so files.
The script will copy all files to the device and build all tests and execute
them.</p>
<p>The test should stop if one of the tests failed.</p>
<p>Example:</p>
<p>Follow the <a href="https://www.tensorflow.org/lite/performance/hexagon_delegate">Instructions</a>
and download the hexagon_nn_skel and extract the files.
For example if files are extracted in /tmp/hexagon_skel, the sample command.</p>
<p><code>bash tensorflow/lite/delegates/hexagon/builders/tests/run_tests.sh /tmp/hexagon_skel</code></p>