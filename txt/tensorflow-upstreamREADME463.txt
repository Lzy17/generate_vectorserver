<h1>TensorFlow Builds</h1>
<p>This directory contains all the files and setup instructions to run all the
important builds and tests. You can run it yourself!</p>
<h2>Run It Yourself</h2>
<p>You have two options when running TensorFlow tests locally on your
machine. First, using docker, you can run our Continuous Integration
(CI) scripts on tensorflow devel images. The other option is to install
all TensorFlow dependencies on your machine and run the scripts
natively on your system.</p>
<h3>Run TensorFlow CI Scripts using Docker</h3>
<ol>
<li>
<p>Install Docker following the <a href="https://docs.docker.com/engine/installation/">instructions on the docker website</a>.</p>
</li>
<li>
<p>Start a container with one of the devel images here:
    https://hub.docker.com/r/tensorflow/tensorflow/tags/.</p>
</li>
<li>
<p>Based on your choice of the image, pick one of the scripts under
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/ci_build/linux
    and run them from the TensorFlow repository root.</p>
</li>
</ol>
<h3>Run TensorFlow CI Scripts Natively on your Machine</h3>
<ol>
<li>
<p>Follow the instructions at https://www.tensorflow.org/install/source,
    but stop when you get to the section "Configure the installation". You do not
    need to configure the installation to run the CI scripts.</p>
</li>
<li>
<p>Pick the appropriate OS and python version you have installed,
    and run the script under tensorflow/tools/ci_build/<OS>.</p>
</li>
</ol>
<h2>TensorFlow Continuous Integration</h2>
<p>To verify that new changes don’t break TensorFlow, we run builds and
tests on either <a href="https://jenkins-ci.org/">Jenkins</a> or a CI system
internal to Google.</p>
<p>We can trigger builds and tests on updates to master or on each pull
request. Contact one of the repository maintainers to trigger builds
on your pull request.</p>
<h3>View CI Results</h3>
<p>The Pull Request will show if the change passed or failed the checks.</p>
<p>From the pull request, click <strong>Show all checks</strong> to see the list of builds
and tests. Click on <strong>Details</strong> to see the results from Jenkins or the internal
CI system.</p>
<p>Results from Jenkins are displayed in the Jenkins UI. For more information,
see the <a href="https://jenkins.io/doc/">Jenkins documentation</a>.</p>
<p>Results from the internal CI system are displayed in the Build Status UI. In
this UI, to see the logs for a failed build:</p>
<ul>
<li>
<p>Click on the <strong>INVOCATION LOG</strong> tab to see the invocation log.</p>
</li>
<li>
<p>Click on the <strong>ARTIFACTS</strong> tab to see a list of all artifacts, including logs.</p>
</li>
<li>
<p>Individual test logs may be available. To see these logs, from the <strong>TARGETS</strong>
    tab, click on the failed target. Then, click on the <strong>TARGET LOG</strong> tab to see
    its test log.</p>
<p>If you’re looking at target that is sharded or a test that is flaky, then
the build tool divided the target into multiple shards or ran the test
multiple times. Each test log is specific to the shard, run, and attempt.
To see a specific log:</p>
<ol>
<li>
<p>Click on the log icon that is on the right next to the shard, run,
    and attempt number.</p>
</li>
<li>
<p>In the grid that appears on the right, click on the specific shard,
    run, and attempt to view its log. You can also type the desired shard,
    run, or attempt number in the field above its grid.</p>
</li>
</ol>
</li>
</ul>
<h3>Third party TensorFlow CI</h3>
<h4><a href="https://www.mellanox.com/">Mellanox</a> TensorFlow CI</h4>
<h5>How to start CI</h5>
<ul>
<li>Submit special pull request (PR) comment to trigger CI: <strong>bot:mlx:test</strong></li>
<li>Test session is run automatically.</li>
<li>Test results and artifacts (log files) are reported via PR comments</li>
</ul>
<h5>CI Steps</h5>
<p>CI includes the following steps: * Build TensorFlow (GPU version) * Run
TensorFlow tests: *
<a href="https://github.com/tensorflow/benchmarks/blob/master/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py">TF CNN benchmarks</a>
(TensorFlow 1.13 and less) *
<a href="https://github.com/tensorflow/models/tree/master/official/r1/resnet">TF models</a>
(TensorFlow 2.0): ResNet, synthetic data, NCCL, multi_worker_mirrored
distributed strategy</p>
<h5>Test Environment</h5>
<p>CI is run in the Mellanox lab on a 2-node cluster with the following parameters:
* Hardware * IB: 1x ConnectX-6 HCA (connected to Mellanox Quantum™ HDR switch) *
GPU: 1x Nvidia Tesla K40m * Software * Ubuntu 16.04.6 * Internal stable
<a href="https://www.mellanox.com/page/products_dyn?product_family=26">MLNX_OFED</a>,
<a href="https://www.mellanox.com/page/hpcx_overview">HPC-X™</a> and
<a href="https://www.mellanox.com/page/products_dyn?product_family=261&amp;mtag=sharp">SHARP™</a>
versions</p>
<h5>Support (Mellanox)</h5>
<p>With any questions/suggestions or in case of issues contact
<a href="mailto:artemry@mellanox.com">Artem Ryabov</a>.</p>