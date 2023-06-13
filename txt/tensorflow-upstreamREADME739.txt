<h1>Compile a StableHLO program with XLA</h1>
<p>This tutorial and the code in this directory shows how to write a simple
StableHLO program and then compile it with XLA. The purpose is simply to
show how XLA can injest a StableHLO program and produce an executable
that's compatible with the local device. As such, the program is very
simple: $\alpha x+y$ ("axpy").</p>
<p>The process includes just a few steps:</p>
<ol>
<li>Construct a StableHLO program using the StableHLO dialect.</li>
<li>Tell XLA to create a "computation" based on this program. In this example,
    we will use PjRt (Pretty much just another Runtime) to achieve that.</li>
<li>Run the compiled executable with some inputs to compute results.</li>
</ol>
<p>All the code is already provided in this directory, which you can build and
run using the steps at the end of this page.</p>
<h2>1. Create the StableHLO program</h2>
<p>We'll define the computation axpy as a StableHLO program, using an
<a href="https://mlir.llvm.org/">MLIR</a> file in the
<a href="https://github.com/openxla/stablehlo">StableHLO</a> dialect.</p>
<p>It can be helpful to consider the computation as a graph, where each node is an
operation (an "op" or "HLO" which means "high-level operation") and the graph
edges are the data flow between operations. So the graph for axpy looks like
this:</p>
<p><code>mermaid
graph TD
    p0(alpha f32) --&gt; mul(Multiply 4xf32)
    p1(x 4xf32) --&gt; mul --&gt; add(Add 4xf32)
    p2(y 4xf32) --&gt; add</code></p>
<p>And here's how we define the program using MLIR (in the StableHLO dialect):</p>
<p><code>mlir
func.func @main(
  %alpha: tensor&lt;f32&gt;, %x: tensor&lt;4xf32&gt;, %y: tensor&lt;4xf32&gt;
) -&gt; tensor&lt;4xf32&gt; {
  %0 = stablehlo.broadcast_in_dim %alpha, dims = []
    : (tensor&lt;f32&gt;) -&gt; tensor&lt;4xf32&gt;
  %1 = stablehlo.multiply %0, %x : tensor&lt;4xf32&gt;
  %2 = stablehlo.add %1, %y : tensor&lt;4xf32&gt;
  func.return %2: tensor&lt;4xf32&gt;
}</code></p>
<p>This code is in <a href="stablehlo_axpy.mlir"><code>stablehlo_axpy.mlir</code></a>.</p>
<p><strong>Note:</strong> StableHLO expresses broadcasting explicitly, so we use
<code>"stablehlo.broadcast_in_dim"</code> to broadcast our scalar to a rank-1 tensor.</p>
<h2>2. Compile the StableHLO program</h2>
<p>Our program for this tutorial is set up as a test in
<a href="stablehlo_compile_test.cc"><code>stablehlo_compile_test.cc</code></a>. In this file,
you'll see that we first set up a <code>PjRtStreamExecutorClient</code> that
allows us to compile our StableHLO program:</p>
<p>```c++
// Setup client
LocalClient* local_client = xla::ClientLibrary::LocalClientOrDie();</p>
<p>// Retrieve the "platform" we intend to execute the computation on. The
// concept of "platform" in XLA abstracts entirely everything need to
// interact with some hardware (compiler, runtime, etc.). New HW vendor
// plugs into XLA by registering a new platform with a different string
// key. For example for an Nvidia GPU change the following to:
//   PlatformUtil::GetPlatform("CUDA"));
TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                        PlatformUtil::GetPlatform("cpu"));
se::StreamExecutorConfig config;
config.ordinal = 0;
TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                        platform-&gt;GetExecutor(config));</p>
<p>// LocalDeviceState and PjRtStreamExecutorDevice describes the state of a
// device which can do computation or transfer buffers. Could represent a GPU
// or accelerator, but we'll use the CPU for this example.
auto device_state = std::make_unique<LocalDeviceState>(
    executor, local_client, LocalDeviceState::kSynchronous,
    /<em>max_inflight_computations=</em>/32,
    /<em>allow_event_reuse=</em>/false, /<em>use_callback_stream=</em>/false);
auto device = std::make_unique<PjRtStreamExecutorDevice>(
    0, std::move(device_state), "cpu");
std::vector<std::unique_ptr\<PjRtStreamExecutorDevice>> devices;
devices.emplace_back(std::move(device));</p>
<p>// The PjRtStreamExecutorClient will allow us to compile and execute
// computations on the device we just configured.
auto pjrt_se_client = PjRtStreamExecutorClient(
    "cpu", local_client, std::move(devices), /<em>process_index=</em>/0,
    /<em>allocator=</em>/nullptr, /<em>host_memory_allocator=</em>/nullptr,
    /<em>should_stage_host_to_device_transfers=</em>/false,
    /<em>gpu_run_options=</em>/nullptr);
```</p>
<p>Then we read the StableHLO program from our MLIR file into a string:</p>
<p>```c++
// Read StableHLO program to string
std::string program_path = tsl::io::JoinPath(
    tsl::testing::XlaSrcRoot(), "examples", "axpy", "stablehlo_axpy.mlir");
std::string program_string;</p>
<p>TF_ASSERT_OK(
    tsl::ReadFileToString(tsl::Env::Default(), program_path, &amp;program_string));
```</p>
<p>In order to parse the StableHLO program, we must first register the appropriate
MLIR dialects:</p>
<p>```c++
// Register MLIR dialects necessary to parse our program. In our case this is
// just the Func dialect and StableHLO.
mlir::DialectRegistry dialects;
dialects.insert<mlir::func::FuncDialect>();
mlir::stablehlo::registerAllDialects(dialects);</p>
<p>// Parse StableHLO program.
auto ctx = std::make_unique<mlir::MLIRContext>(dialects);
mlir::OwningOpRef<mlir::ModuleOp> program =
    mlir::parseSourceString<mlir::ModuleOp>(program_string, ctx.get());
```</p>
<p>Now that we've set up our client and parsed the StableHLO program we can
compile it to an executable:</p>
<p><code>c++
// Use our client to compile our StableHLO program to an executable.
TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr&lt;PjRtLoadedExecutable&gt; executable,
                        pjrt_se_client.Compile(*program, CompileOptions{}));</code></p>
<h2>3. Execute the computation</h2>
<p>Finally, in <a href="stablehlo_compile_test.cc"><code>stablehlo_compile_test.cc</code></a>,
we can feed the executable some inputs for the three arguments and
compute the results:</p>
<p>```c++
// Create inputs to our computation.
auto alpha_literal = xla::LiteralUtil::CreateR0<float>(3.14f);
auto x_literal = xla::LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
auto y_literal =
    xla::LiteralUtil::CreateR1<float>({10.5f, 20.5f, 30.5f, 40.5f});</p>
<p>// Get the host device.
PjRtDevice* cpu = pjrt_se_client.devices()[0];</p>
<p>// Transfer our literals to buffers. If we were using a GPU, these buffers
// would correspond to device memory.
TF_ASSERT_OK_AND_ASSIGN(
    std::unique_ptr<PjRtBuffer> alpha,
    pjrt_se_client.BufferFromHostLiteral(alpha_literal, cpu));
TF_ASSERT_OK_AND_ASSIGN(
    std::unique_ptr<PjRtBuffer> x,
    pjrt_se_client.BufferFromHostLiteral(x_literal, cpu));
TF_ASSERT_OK_AND_ASSIGN(
    std::unique_ptr<PjRtBuffer> y,
    pjrt_se_client.BufferFromHostLiteral(y_literal, cpu));</p>
<p>// Do our computation.
TF_ASSERT_OK_AND_ASSIGN(
    std::vector<std::vector\<std::unique_ptr\<PjRtBuffer>>> axpy_result,
    executable-&gt;Execute({{alpha.get(), x.get(), y.get()}}, /<em>options=</em>/{}));</p>
<p>// Convert result buffer back to literal.
TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> axpy_result_literal,
                        axpy_result[0][0]-&gt;ToLiteralSync());</p>
<p>// Check to make sure that our results match what we expect.
xla::LiteralTestUtil::ExpectR1Near<float>({13.64f, 26.78f, 39.92f, 53.06f},
                                          *axpy_result_literal,
                                          xla::ErrorSpec(0.01f));
```</p>
<h2>4. Build and run the code</h2>
<p>You can build and run this example as follows using
<a href="https://github.com/bazelbuild/bazelisk#readme">Bazelisk</a> or
<a href="https://bazel.build/">Bazel</a> (run from within <code>xla/examples/axpy/</code>):</p>
<p><code>sh
bazelisk test :stablehlo_compile_test --test_output=all --nocheck_visibility</code></p>
<p>Sample output from the test should look like this:</p>
<p>```sh
==================== Test output for //xla/examples/axpy:stablehlo_compile_test:
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from StableHloAxpyTest
[ RUN      ] StableHloAxpyTest.LoadAndRunCpuExecutable
Loaded StableHLO program from xla/examples/axpy/stablehlo_axpy.mlir:
func.func @main(
  %alpha: tensor<f32>, %x: tensor&lt;4xf32&gt;, %y: tensor&lt;4xf32&gt;
) -&gt; tensor&lt;4xf32&gt; {
  %0 = stablehlo.broadcast_in_dim %alpha, dims = []
    : (tensor<f32>) -&gt; tensor&lt;4xf32&gt;
  %1 = stablehlo.multiply %0, %x : tensor&lt;4xf32&gt;
  %2 = stablehlo.add %1, %y : tensor&lt;4xf32&gt;
  func.return %2: tensor&lt;4xf32&gt;
}</p>
<p>Computation inputs:
        alpha:f32[] 3.14
        x:f32[4] {1, 2, 3, 4}
        y:f32[4] {10.5, 20.5, 30.5, 40.5}
Computation output: f32[4] {13.64, 26.78, 39.920002, 53.06}
[       OK ] StableHloAxpyTest.LoadAndRunCpuExecutable (264 ms)
[----------] 1 test from StableHloAxpyTest (264 ms total)</p>
<p>[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (264 ms total)
[  PASSED  ] 1 test.
```</p>