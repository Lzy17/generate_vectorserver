<h1>MLIR Replay tool</h1>
<p>This tool is mainly intended for helping debug miscompiles. It takes as inputs
an HLO snapshot proto with input tensors and a compiler trace proto with the
state of the IR after each pass.</p>
<p>This tool is built on top of
<a href="https://github.com/tensorflow/mlir-hlo/tree/master/tools/mlir_interpreter/">mlir-interpreter</a>.</p>
<p>Example usage:</p>
<p>```</p>
<h1>Run a JAX test with debug flags enabled:</h1>
<p>$ bazel test :some_jax_test --compilation_mode=opt \
  --test_env=XLA_FLAGS="--xla_cpu_use_xla_runtime --xla_dump_to=/tmp/test-dump --xla_dump_hlo_snapshots" \
  --test_filter=SomeSpecific.TestCase \
  --test_sharding_strategy=disabled --test_strategy=local</p>
<h1>JAX tends to compile many modules, so first check which one is broken:</h1>
<p>./mlir_replay \
  --mlir-compilation-trace-dir=/tmp/test-dump</p>
<p>Failures for /tmp/test-dump/module_1234.jit_something.mlir-trace.pb:
  Result mismatch for /tmp/test-dump/module_1234.jit_something.snapshot.56.pb: TensorOrMemref&lt;3xi32&gt;: [1, 2, 3] != TensorOrMemref&lt;3xi32&gt;: [1, 1, 1]
  run :mlir_replay -- --mlir-compilation-trace=/tmp/test-dump/module_1234.jit_something.mlir-trace.pb --hlo-snapshot=/tmp/test-dump/module_1234.jit_something.snapshot.56.pb --print-changes-only --stop-after-first-failure
```</p>
<p>There may be multiple failing modules. You can run the provided command to
replay a particular one:</p>
<p>```</p>
<h1>Run the IR after each pass. Note that JAX typically compiles many modules, so</h1>
<h1>you may have check more than one.</h1>
<h1>There is one .mlir-trace.pb file per module (containing the intermediate IR)</h1>
<h1>and one .snapshot.pb file per execution (containing the inputs and outputs).</h1>
<p>$ ./mlir_replay \
  --mlir-compilation-trace=/tmp/test-dump/module_1234.jit_something.mlir-trace.pb \
  --hlo-snapshot=/tmp/test-dump/module_1234.jit_something.snapshot.56.pb \
  --print-changes-only --stop-after-first-failure
Running IR after APass
Results: [1, 2, 3]</p>
<p>Running IR after BPass
Running IR after CPass
Running IR after BrokenPass
Results: [1, 1, 1]
```</p>