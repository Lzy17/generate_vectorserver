<h2>TestApp</h2>
<p>The TestApp is currently being used as a dummy app by Circle CI for nightly jobs. The challenge comes when testing the arm64 build as we don't have a way to code-sign our TestApp. This is where Fastlane came to rescue. <a href="https://fastlane.tools/">Fastlane</a> is a trendy automation tool for building and managing iOS applications. It also works seamlessly with Circle CI. We are going to leverage the <code>import_certificate</code> action, which can install developer certificates on CI machines. See <code>Fastfile</code> for more details.</p>
<p>For simulator build, we run unit tests as the last step of our CI workflow. Those unit tests can also be run manually via the <code>fastlane scan</code> command.</p>
<h2>Run Simulator Test Locally</h2>
<p>Follow these steps if you want to run the test locally.</p>
<ol>
<li>
<p>Checkout PyTorch repo including all submodules</p>
</li>
<li>
<p>Build PyTorch for ios
<code>USE_COREML_DELEGATE=1 IOS_PLATFORM=SIMULATOR ./scripts/build_ios.sh</code></p>
</li>
<li>
<p>Generate on-the-fly test models
<code>python test/mobile/model_test/gen_test_model.py ios-test</code>
You need to install regular PyTorch on your local machine to run this script.
Check https://github.com/pytorch/pytorch/tree/master/test/mobile/model_test#diagnose-failed-test to learn more.</p>
</li>
<li>
<p>Create XCode project (for lite interpreter)
<code>cd ios/TestApp/benchmark
ruby setup.rb --lite 1</code></p>
</li>
<li>
<p>Open the generated TestApp/TestApp.xcodeproj in XCode and run simulator test.</p>
</li>
</ol>
<h2>Re-generate All Test Models</h2>
<ol>
<li>
<p>Make sure PyTorch (not PyTorch for iOS) is installed
See https://pytorch.org/get-started/locally/</p>
</li>
<li>
<p>Re-generate models for operator test
<code>python test/mobile/model_test/gen_test_model.py ios
python test/mobile/model_test/gen_test_model.py ios-test</code></p>
</li>
<li>
<p>Re-generate Core ML model
<code>cd ios/TestApp/benchmark; python coreml_backend.py</code></p>
</li>
</ol>
<h2>Debug Test Failures</h2>
<p>Make sure all models are generated. See https://github.com/pytorch/pytorch/tree/master/test/mobile/model_test to learn more.</p>
<p>There's no debug information in simulator test (project TestAppTests). You can copy the failed test code to
TestApp/TestApp/ViewController.mm and debug in the main TestApp.</p>
<h3>Benchmark</h3>
<p>The benchmark folder contains two scripts that help you setup the benchmark project. The <code>setup.rb</code> does the heavy-lifting jobs of setting up the XCode project, whereas the <code>trace_model.py</code> is a Python script that you can tweak to generate your model for benchmarking. Simply follow the steps below to setup the project</p>
<ol>
<li>In the PyTorch root directory, run <code>IOS_ARCH=arm64 ./scripts/build_ios.sh</code> to generate the custom build from <strong>Master</strong> branch</li>
<li>Navigate to the <code>benchmark</code> folder, run <code>python trace_model.py</code> to generate your model.</li>
<li>In the same directory, open <code>config.json</code>. Those are the input parameters you can tweak.</li>
<li>Again, in the same directory, run <code>ruby setup.rb</code> to setup the XCode project.</li>
<li>Open the <code>TestApp.xcodeproj</code>, you're ready to go.</li>
</ol>
<p>The benchmark code is written in C++, you can use <code>UI_LOG</code> to visualize the log. See <code>benchmark.mm</code> for more details.</p>