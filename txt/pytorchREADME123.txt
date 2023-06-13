<h1>PyTorch/Caffe2 Operator Micro-benchmarks</h1>
<p>This benchmark suite provides a systemic way to measure the performance of operators for a wide range of inputs. The generated benchmark data fully characterized the performance of an operator in terms of execution time and the efficiency of the PyTorch/Caffe2 frameworks used.</p>
<h2>Features</h2>
<p>Key Features:</p>
<p>1. Language used: Python</p>
<p>2. Supported Frameworks: PyTorch and Caffe2</p>
<p>3. Supported PyTorch mode: eager and JIT</p>
<p>4. Input shapes: user-defined shapes, randomly generated shapes</p>
<h2>Getting Started</h2>
<h2>Initial Setup</h2>
<p>The instruction below installs a cpp_extension for PyTorch and it is required to run the benchmark suite.
<code>$ cd pt_extension
$ python setup.py install</code></p>
<h2>How to run the benchmarks:</h2>
<p>Run <code>torch.add</code> benchmark:
<code>$ cd pytorch/benchmarks/operator_benchmark
$ python -m pt.add_test --omp-num-threads 1 --mkl-num-threads 1</code>
Note: we set the number of OpenMP and MKL threads both to 1. If you want to benchmark operators with multithreading (intra-op parallelism), use the <code>--omp-num-threads</code> and <code>--mkl-num-threads</code> flags.</p>
<p>List all the supported tests:
<code>$ python -m pt.add_test --list-tests</code></p>
<p>Filter and run a test (use <code>add_M8_N16_K32</code> as an example):
<code>$ python -m pt.add_test --test-name add_K32_M8_N1
--omp-num-threads 1 --mkl-num-threads 1</code></p>
<p>Run all the supported benchmarks:
<code>$ python -m benchmark_all_test</code></p>
<h2>Code to support <code>torch.add</code> in the benchmark</h2>
<p>The following example shows the code to support <code>torch.add</code> with 27 different tests. In the subpages of this wiki, we'll step through the complete flow of adding PyTorch and Caffe2 operators to the benchmark suite. Existing benchmarks for operators are in <code>pt</code> and <code>c2</code> directories and we highly recommend putting your new operators in those locations.</p>
<p>```
add_short_configs = op_bench.cross_product_configs(
    M=[8, 64, 128],
    N=range(2, 10, 3),
    K=[2 ** x for x in range(0, 3)],
    tags=["short"]
)</p>
<p>class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.inputs = {
            "input_one": torch.rand(M, N, K, device=device, requires_grad=self.auto_set()),
            "input_two": torch.rand(M, N, K, device=device, requires_grad=self.auto_set())
        }
        self.set_module_name("add")</p>
<pre><code>def forward(self, input_one, input_two):
    return torch.add(input_one, input_two)
</code></pre>
<p>op_bench.generate_pt_test(add_short_configs, AddBenchmark)
```</p>
<h2>Output and Command Line Control of the Benchmark</h2>
<p>The output is intended to be a human readable format. Here is an example output for <code>torch.add</code>:
```</p>
<h1>----------------------------------------</h1>
<h1>PyTorch/Caffe2 Operator Micro-benchmarks</h1>
<h1>----------------------------------------</h1>
<h1>Tag : short</h1>
<h1>Benchmarking PyTorch: add</h1>
<h1>Mode: Eager</h1>
<h1>Name: add_M8_N16_K32</h1>
<h1>Input: M: 8, N: 16, K: 32</h1>
<p>Forward Execution Time (us) : 6.651</p>
<h1>Benchmarking PyTorch: add</h1>
<h1>Mode: Eager</h1>
<h1>Name: add_M16_N16_K64</h1>
<h1>Input: M: 16, N: 16, K: 64</h1>
<p>Forward Execution Time (us) : 11.976</p>
<h1>Benchmarking PyTorch: add</h1>
<h1>Mode: Eager</h1>
<h1>Name: add_M64_N64_K128</h1>
<h1>Input: M: 64, N: 64, K: 128</h1>
<p>Forward Execution Time (us) : 222.370
<code>``
At a high level, the output includes the execution time of</code>torch.add` with three different inputs. Let's look at each line in detail:</p>
<p>1. <code>Tag: short</code> tags a group of inputs. For each operator, you could be interested in a large number of inputs, but you may not always want to run all the inputs. <code>Tag</code> allows you to only run some of the inputs. Most of the inputs to operators being supported in the benchmark are grouped using two tags. One group is tagged with <code>short</code> which stores some commonly used shapes. The other group is tagged with <code>long</code> which stores many random inputs to have better coverage compared with <code>short</code>.</p>
<p>2. <code>Benchmarking PyTorch: Add</code> shows name of the operator being benchmarked.</p>
<p>3. <code>Mode: Eager</code> shows that PyTorch eager mode is here.</p>
<p>4. <code>Name: add_M8_N16_K32</code> is the name of the test and it can be used to filter tests.</p>
<p>5. <code>Input: M: 8, N: 16, K: 32</code> shows inputs to the operator.</p>
<p>6. <code>Forward Execution Time (us) : 6.651</code> reports the execution time of an operator in microseconds.</p>
<h3>Command-Line Control</h3>
<p>You can control all the aspects of the benchmark suite through the command-line. Please find details of those arguments by running the following command or look into <code>benchmark_runner.py</code>.
<code>$ python benchmark_runner.py --help</code></p>
<p>Run all the supported benchmarks:
<code>$ python -m benchmark_all_test --omp-num-threads 1 --mkl-num-threads 1</code></p>
<p>List all the supported operators:
<code>$ python -m benchmark_all_test --list-ops</code></p>
<p>List all the supported tests:
<code>$ python -m benchmark_all_test --list-tests</code></p>
<p>Filter and run an operator (use add as an example):
<code>$ python -m benchmark_all_test --operators add --omp-num-threads 1 --mkl-num-threads 1</code>
Note: this filter is based on the operator name rather than the file name.</p>
<p>Run torch.add benchmark with tag 'long':
<code>$ python -m pt.add_test --tag-filter long</code></p>
<h2>Adding New Operators to the Benchmark Suite</h2>
<p>In the previous sections, we gave several examples to show how to run the already available operators in the benchmark suite. In the following sections, we'll step through the complete flow of adding PyTorch and Caffe2 operators to the benchmark suite. Existing benchmarks for operators are in <code>pt</code> and <code>c2</code> directories and we highly recommend putting your new operators in those directories as well.</p>
<h3>Add a New PyTorch Operator</h3>
<p>Let's say you want to measure the execution time of the following operator:
<code>C = torch.add(A, B) # Shape of A and B is [M, N, K]</code>
The code below shows how to add it to the benchmark suite. Let's go over the example line by line.
```
import operator_benchmark as op_bench
import torch</p>
<p>add_long_configs = op_bench.cross_product_configs(
    M=[8, 64, 128],
    N=range(2, 10, 3),
    K=[2 ** x for x in range(0, 3)],
    tags=["long"]
)</p>
<p>add_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [8, 16, 32],
        [16, 16, 64],
        [64, 64, 128],
    ],
    tags=["short"],
)</p>
<p>class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.inputs = {
            "input_one": torch.rand(M, N, K, device=device, requires_grad=self.auto_set()),
            "input_two": torch.rand(M, N, K, device=device, requires_grad=self.auto_set())
        }
        self.set_module_name("add")</p>
<pre><code>def forward(self, input_one, input_two):
    return torch.add(input_one, input_two)
</code></pre>
<p>op_bench.generate_pt_test(add_long_configs + add_short_configs, AddBenchmark)</p>
<p>if <strong>name</strong> == "<strong>main</strong>":
    op_bench.benchmark_runner.main()
```</p>
<h4>Part 1. Specify Inputs to Operators</h4>
<p>For the <code>torch.add</code> operator, we would like to make sure it delivers good performance with input tensors which are of small, medium and large sizes. We have introduced two helper functions for users to easily generate a combination of inputs.
```</p>
<h1>Generate list configurations that will be used for benchmark experiments</h1>
<p>add_long_configs = op_bench.cross_product_configs(
    M=[8, 64, 128],
    N=range(2, 10, 3),
    K=[2 ** x for x in range(0, 3)],
    tags=["long"]
)</p>
<p>add_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [8, 16, 32],
        [16, 16, 64],
        [64, 64, 128],
    ],
    tags=["short"],
)
```
Let's look at it in detail:</p>
<p>1. <code>op_bench.config_list</code> is a helper function which specifies a list of inputs to operators. It takes three parameters which are <code>attrs_names, attrs, and tags</code>, all of them are python lists. <code>attr_names</code> stores the names of the inputs. <code>attrs</code> stores the real value of each input. In this example, three different inputs will be returned which are: <code>M=8, N=16, K=32; M=16, N=16, K=64; M=64, N=64, K=128</code>.</p>
<p>2. <code>op_bench.cross_product_configs</code> is another helper function to generate a cartesian product of the inputs. Each input is specified in a python list. In this example, the helper method will return a combination of 27 (len(M) * len(N) * len(K)) inputs.</p>
<h4>Part 2. Create Tensors and Add Computation</h4>
<p>After inputs are provided, we now look at adding the computation of an operator. Adding a new operator requires implementing a new <code>TorchBenchmarkBase</code> subclass. Every new class is required to implement 2 methods:
* <code>init</code> is used to create tensors based on the inputs we provided before. In this example, the parameters to <code>init</code> are <code>M, N, and K</code> which have been specified in the input configuration. <code>init</code> also packed all the needed inputs together into a dictionary <code>self.inputs</code> which will be provided to <code>forward</code> as arguments for running the benchmark.
* <code>forward</code> includes the operator to be tested and the computation based on the created tensors in <code>init</code>. Apart from <code>self</code>, the order of the arguments must match the entries specified in <code>self.inputs</code>.</p>
<p>The example below shows the code for <code>torch.add</code>:
```</p>
<h1>Given one set of M, N, K, the init method creates input tensors based on</h1>
<h1>that. The forward method does torch.add calculation on those input tensors.</h1>
<p>class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        # this is the method where you need to create tensors
        # M, N, and K can be in different order, but they must match with
        # names in the configs.
        self.inputs = {
            "input_one": torch.rand(M, N, K, device=device, requires_grad=self.auto_set()),
            "input_two": torch.rand(M, N, K, device=device, requires_grad=self.auto_set())
        }
        self.set_module_name("add")</p>
<pre><code>def forward(self, input_one, input_two):
    # this is the method to have operator and do computation
    return torch.add(input_one, input_two)
</code></pre>
<p>```</p>
<h4>Part 3. Register Tests With the Benchmark Suite</h4>
<p>After we have inputs and the benchmark class, it's time to register them with our benchmark suite. Here is how it looks like:
<code>op_bench.generate_pt_test(add_long_configs + add_short_configs, AddBenchmark)</code>
<code>generate_pt_test</code> takes two parameters which are inputs configs and the benchmark class.</p>
<h4>Part 4. Run the Registered Tests</h4>
<p>To run the benchmark, we use the main method in <code>benchmark_runner</code> module.
<code>if __name__ == "__main__":
    op_bench.benchmark_runner.main()</code>
That's it. You just added a new operator to the benchmark suite!</p>
<h3>Add a New Caffe2 Operator</h3>
<p>The steps to add a new Caffe2 operator is the same as that for a PyTorch operator. The code below shows how to add Caffe2 <code>Add</code> operator:
```
import operator_benchmark as op_bench
from caffe2.python import core</p>
<p>add_long_configs = op_bench.cross_product_configs(
    M=[8, 64, 128],
    N=range(2, 10, 3),
    K=[2 ** x for x in range(0, 3)],
    tags=["long"]
)</p>
<p>add_short_configs = op_bench.config_list(
    attrs=[
        [8, 16, 32],
        [16, 16, 64],
        [64, 64, 128],
    ],
    attr_names=["M", "N", "K"],
    tags=["short"],
)</p>
<p>class AddBenchmark(op_bench.Caffe2BenchmarkBase):</p>
<pre><code>def init(self, M, N, K):
    self.input_one = self.tensor(M, N, K)
    self.input_two = self.tensor(M, N, K)
    self.output = self.tensor(M, N, K)
    self.set_module_name("add")

def forward(self):
    op = core.CreateOperator(
        "Add", [self.input_one, self.input_two], self.output, **self.args
    )

    return op
</code></pre>
<p>op_bench.generate_c2_test(add_long_configs + add_short_configs, AddBenchmark)</p>
<p>if <strong>name</strong> == "<strong>main</strong>":
    op_bench.benchmark_runner.main()
<code>``
There are two things worth mentioning in this code:
*</code>self.tensor<code>is a helper function which takes shapes and returns a Caffe2 blob. It is designed to make the tensor creation step easier compared to the standard Caffe2 way.
*</code>generate_c2_test` is used to register Caffe2 tests with the benchmark.</p>
<h3>Add a List of Operators</h3>
<p>In the previous sections, we introduced the steps required to add a single operator to the benchmark suite. There are scenarios where you want to extend the benchmark suite with a list of operators which can share the same inputs. For example, to benchmark <code>abs</code> and <code>acos</code> operators, you can use the same set of inputs for both.</p>
<p>Let's say we want to benchmark the following operators separately:
<code>C = torch.abs(A) # Shape of A [M, N]
C = torch.acos(A) # Shape of A [M, N]</code>
The following code shows how to do that:
```
import operator_benchmark as op_bench
import torch</p>
<p>unary_ops_configs = op_bench.config_list(
    attrs=[
        [128, 128],
        [256, 256],
        [1024, 1024],
    ],
    attr_names=["M", "N"],
    tags=["short"]
)</p>
<p>unary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["abs", torch.abs],
        ["acos", torch.acos],
    ],
)</p>
<p>class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device, op_func):
        self.inputs = {
            "input": torch.rand(M, N, device=device)
        }
        self.op_func = op_func</p>
<pre><code>def forward(self, input):
    return self.op_func(input)
</code></pre>
<p>op_bench.generate_pt_tests_from_op_list(unary_ops_list, unary_ops_configs, UnaryOpBenchmark)</p>
<p>if <strong>name</strong> == "<strong>main</strong>":
    op_bench.benchmark_runner.main()
```
The inputs to those operators are specified using the same method we went over before. So we just skip it here.</p>
<h4>Part 1. Specify the List of Operators</h4>
<p>To add a list of operators to the benchmark suite, we introduce the <code>op_bench.op_list</code> method which takes two parameters:
* <code>attrs</code> stores the name of the operator and the method to do the real calculation.
* <code>attr_names</code> stores the names of values in attrs.</p>
<p>The example below shows the code to add <code>torch.abs</code> and <code>torch.acos</code> :
<code>unary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["abs", torch.abs],
        ["acos", torch.acos],
    ],
)</code></p>
<h4>Part 2. Create Tensors and Add Computation</h4>
<p>In this example, both operators share the same input so we only need to implement one TorchBenchmarkBase subclass.
Every new subclass is required to implement 3 methods:
* <code>init</code> is used to create tensors and set the operator name and function. In this example, the parameters to <code>init</code> are <code>M</code>, <code>N</code>, and <code>op_func</code> which have been specified in the configurations.
* <code>forward</code> includes the operator to be tested and the computation based on the created tensors in <code>init</code>. Apart from <code>self</code>, the order of the arguments must match the entries specified in <code>self.inputs</code>.
Here is the code for <code>abs</code> and <code>acos</code>:</p>
<p>```
class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device, op_func):
        # The M and N match with the attr_names in the input configuration
        # The op_func matches with the attr_name in the ops configuration
        self.inputs = {
            "input": torch.rand(M, N, device=device)
        }
        self.op_func = op_func</p>
<pre><code>def forward(self, input):
    return self.op_func(input)
</code></pre>
<p>```</p>
<h4>Part 3. Register a List of Operators</h4>
<p>To register multiple operators,  we introduced the <code>generate_pt_tests_from_op_list</code> function which takes three parameters. First, the list of operators. Second,the configs. Third, the benchmark class.
Here is an example:
<code>op_bench.generate_pt_tests_from_op_list(unary_ops_list, unary_ops_configs, UnaryOpBenchmark)</code></p>
<h3>Add Gradient Ops</h3>
<p>In this section, we go over the steps to benchmark the backward path of operators.</p>
<h4>For PyTorch Gradient Ops</h4>
<p>To measure the performance of an operator in its backward path, there are only two changes needed in addition to the steps we covered for the forward path:</p>
<p>1. Specify <code>requires_grad=True</code> when creating the tensor. This is a standard PyTorch way of enabling backward path.</p>
<p>2. Use <code>generate_pt_gradient_test</code> to register the tests.</p>
<p>The example below shows the relevant code for that:
<code>self.input_one = torch.rand(M, N, K, requires_grad=True)
generate_pt_gradient_test(long_configs + short_configs, TorchAddBenchmark)</code></p>
<h4>For Caffe2 Gradient Ops</h4>
<p>To add Caffe2 gradient ops, we need to implement a new backward method in the benchmark class:
```
class AddBenchmark(op_bench.Caffe2BenchmarkBase):</p>
<pre><code>def init(self, M, N, K):
    self.input_one = self.tensor(M, N, K)
    self.input_two = self.tensor(M, N, K)
    self.input_one_grad = self.tensor(M, N, K)
    self.input_two_grad = self.tensor(M, N, K)
    self.output = self.tensor(M, N, K)
    self.set_module_name("add")

def forward(self):
    op = core.CreateOperator(
        "Add", [self.input_one, self.input_two], self.output, **self.args
    )

    return op

def backward(self):
    grad_op = core.CreateOperator(
        "AddGradient",
        [self.output, self.input_one, self.input_two],
        [self.input_one_grad, self.input_two_grad], **self.args
    )

    return grad_op
</code></pre>
<p>op_bench.generate_c2_gradient_test(long_configs + short_configs,AddBenchmark)
<code>``
After the class is implemented, we need to register the tests with</code>generate_c2_gradient_test` function.</p>
<p>This concludes the overview of the operator benchmark suite.</p>