<h1>Code Coverage Tool for Pytorch</h1>
<h2>Overview</h2>
<p>This tool is designed for calculating code coverage for Pytorch project.
It’s an integrated tool. You can use this tool to run and generate both file-level and line-level report for C++ and Python tests. It will also be the tool we use in <em>CircleCI</em> to generate report for each main commit.</p>
<h3>Simple</h3>
<ul>
<li><em>Simple command to run:</em><ul>
<li><code>python oss_coverage.py</code></li>
</ul>
</li>
<li><em>Argument <code>--clean</code> will do all the messy clean up things for you</em></li>
</ul>
<h3>But Powerful</h3>
<ul>
<li><em>Choose your own interested folder</em>:<ul>
<li>Default folder will be good enough in most times</li>
<li>Flexible: you can specify one or more folder(s) that you are interested in</li>
</ul>
</li>
<li><em>Run only the test you want:</em><ul>
<li>By default it will run all the c++ and python tests</li>
<li>Flexible: you can specify one or more test(s) that you want to run</li>
</ul>
</li>
<li><em>Final report:</em><ul>
<li>File-Level: The coverage percentage for each file you are interested in</li>
<li>Line-Level: The coverage details for each line in each file you are interested in</li>
<li>Html-Report (only for <code>gcc</code>): The beautiful HTML report supported by <code>lcov</code>, combine file-level report and line-lever report into a graphical view.</li>
</ul>
</li>
<li><em>More complex but flexible options:</em><ul>
<li>Use different stages like <em>--run, --export, --summary</em> to achieve more flexible functionality</li>
</ul>
</li>
</ul>
<h2>How to use</h2>
<p>This part will introduce about the arguments you can use when run this tool. The arguments are powerful, giving you full flexibility to do different work.
We have two different compilers, <code>gcc</code> and <code>clang</code>, and this tool supports both. But it is recommended to use <code>gcc</code> because it's much faster and use less disk place. The examples will also be divided to two parts, for <code>gcc</code> and <code>clang</code>.</p>
<h2>Preparation</h2>
<p>The first step is to <a href="https://github.com/pytorch/pytorch#from-source">build <em>Pytorch</em> from source</a> with <code>USE_CPP_CODE_COVERAGE</code> option <code>ON</code>. You may also want to set <code>BUILD_TEST</code> option <code>ON</code> to get the test binaries. Besides, if you are under <code>gcc</code> compiler, to get accurate result, it is recommended to also select <code>CMAKE_BUILD_TYPE=Debug</code>.
See: <a href="https://github.com/pytorch/pytorch#adjust-build-options-optional">how to adjust build options</a> for reference. Following is one way to adjust build option:
```</p>
<h1>in build/ folder (all build artifacts must in <code>build/</code> folder)</h1>
<p>cmake .. -DUSE_CPP_CODE_COVERAGE=ON -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Debug
```</p>
<h2>Examples</h2>
<p>The tool will auto-detect compiler type in your operating system, but if you are using another one, you need to specify it. Besides, if you are using <code>clang</code>, <code>llvm</code> tools are required. So the first step is to set some environment value if needed:
```bash</p>
<h1>set compiler type, the default is auto detected, you can check it at the start of log.txt</h1>
<p>export COMPILER_TYPE="CLANG"</p>
<h1>set llvm path for clang, by default is /usr/local/opt/llvm/bin</h1>
<p>export LLVM_TOOL_PATH=...
```</p>
<p>Great, you are ready to run the code coverage tool for the first time! Start from the simple command:
<code>python oss_coverage.py --run-only=atest</code>
This command will run <code>atest</code> binary in <code>build/bin/</code> folder and generate reports over the entire <em>Pytorch</em> folder. You can find the reports in <code>profile/summary</code>. But you may only be interested in the <code>aten</code> folder, in this case, try:
<code>python oss_coverage.py --run-only=atest --interest-only=aten</code>
In <em>Pytorch</em>, <code>c++</code> tests located in <code>build/bin/</code> and <code>python</code> tests located in <code>test/</code>. If you want to run <code>python</code> test, try:
<code>python oss_coverage.py --run-only=test_complex.py</code></p>
<p>You may also want to specify more than one test or interested folder, in this case, try:
<code>python oss_coverage.py --run-only=atest c10_logging_test --interest-only aten/src/Aten c10/core</code>
That it is! With these two simple options, you can customize many different functionality according to your need.
By default, the tool will run all tests in <code>build/bin</code> folder (by running all executable binaries in it) and <code>test/</code> folder (by running <code>run_test.py</code>), and then collect coverage over the entire <em>Pytorch</em> folder. If this is what you want, try:
<em>(Note: It's not recommended to run default all tests in clang, because it will take too much space)</em>
<code>bash
python oss_coverage.py</code></p>
<h3>For more complex arguments and functionalities</h3>
<h4>GCC</h4>
<p>The code coverage with <code>gcc</code> compiler can be divided into 3 step:
1. run the tests: <code>--run</code>
2. run <code>gcov</code> to get json report: <code>--export</code>
3. summarize it to human readable file report and line report: <code>--summary</code></p>
<p>By default all steps will be run, but you can specify only run one of them. Following is some usage scenario:</p>
<p><strong>1. Interested in different folder</strong>
<code>—summary</code> is useful when you have different interested folder. For example,
```bash</p>
<h1>after run this command</h1>
<p>python oss_coverage.py --run-only=atest --interest-only=aten</p>
<h1>you may then want to learn atest's coverage over c10, instead of running the test again, you can:</h1>
<p>python oss_coverage.py --run-only=atest --interest-only=c10 --summary
```</p>
<p><strong>2. Run tests yourself</strong>
When you are developing a new feature, you may first run the tests yourself to make sure the implementation is all right and then want to learn its coverage. But sometimes the test take very long time and you don't want to wait to run it again when doing code coverage. In this case, you can use these arguments to accelerate your development (make sure you build pytorch with the coverage option!):
```</p>
<h1>run tests when you are developing a new feature, assume the test is <code>test_nn.py</code></h1>
<p>python oss_coverage.py --run-only=test_nn.py</p>
<h1>or you can run it yourself</h1>
<p>cd test/ &amp;&amp; python test_nn.py</p>
<h1>then you want to learn about code coverage, you can just run:</h1>
<p>python oss_coverage.py --run-only=test_nn.py --export --summary
```</p>
<h3>CLANG</h3>
<p>The steps for <code>clang</code> is very similar to <code>gcc</code>, but the export stage is divided into two step:
1. run the tests: <code>--run</code>
2. run <code>gcov</code> to get json report: <code>--merge</code> <code>--export</code>
3. summarize it to human readable file report and line report: <code>--summary</code></p>
<p>Therefore, just replace <code>--export</code> in <code>gcc</code> examples with <code>--merge</code> and <code>--export</code>, you will find it work!</p>
<h2>Reference</h2>
<p>For <code>gcc</code>
* See about how to invoke <code>gcov</code>, read <a href="https://gcc.gnu.org/onlinedocs/gcc/Invoking-Gcov.html#Invoking-Gcov">Invoking gcov</a> will be helpful</p>
<p>For <code>clang</code>
* If you are not familiar with the procedure of generating code coverage report by using <code>clang</code>, read <a href="https://clang.llvm.org/docs/SourceBasedCodeCoverage.html">Source-based Code Coverage</a> will be helpful.</p>