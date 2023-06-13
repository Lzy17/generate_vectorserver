<h1>Modular Benchmarking Components:</h1>
<p>NOTE: These components are currently work in progress.</p>
<h2>Timer</h2>
<p>This class is modeled on the <code>timeit.Timer</code> API, but with PyTorch specific
facilities which make it more suitable for benchmarking kernels. These fall
into two broad categories:</p>
<h3>Managing 'gotchas':</h3>
<p><code>Timer</code> will invoke <code>torch.cuda.synchronize()</code> if applicable, control the
  number of torch threads, add a warmup, and warn if a measurement appears
  suspect or downright unreliable.</p>
<h3>Integration and better measurement:</h3>
<p><code>Timer</code>, while modeled after the <code>timeit</code> analog, uses a slightly different
  API from <code>timeit.Timer</code>.</p>
<ul>
<li>
<p>The constructor accepts additional metadata and timing methods return
  a <code>Measurement</code> class rather than a float. This <code>Measurement</code> class is
  serializable and allows many examples to be grouped and interpreted.
  (See <code>Compare</code> for more details.)</p>
</li>
<li>
<p><code>Timer</code> implements the <code>blocked_autorange</code> function which is a
  mixture of <code>timeit.Timer.repeat</code> and <code>timeit.Timer.autorange</code>. This function
  selects and appropriate number and runs for a roughly fixed amount of time
  (like <code>autorange</code>), but is less wasteful than <code>autorange</code> which discards
  ~75% of measurements. It runs many times, similar to <code>repeat</code>, and returns
  a <code>Measurement</code> containing all of the run results.</p>
</li>
</ul>
<h2>Compare</h2>
<p><code>Compare</code> takes a list of <code>Measurement</code>s in its constructor, and displays them
as a formatted table for easier analysis. Identical measurements will be
merged, which allows <code>Compare</code> to process replicate measurements. Several
convenience methods are also provided to truncate displayed values based on
the number of significant figures and color code measurements to highlight
performance differences. Grouping and layout is based on metadata passed to
<code>Timer</code>:
* <code>label</code>: This is a top level description. (e.g. <code>add</code>, or <code>multiply</code>) one
table will be generated per unique label.</p>
<ul>
<li>
<p><code>sub_label</code>: This is the label for a given configuration. Multiple statements
may be logically equivalent differ in implementation. Assigning separate
sub_labels will result in a row per sub_label. If a sublabel is not provided,
<code>stmt</code> is used instead. Statistics (such as computing the fastest
implementation) are use all sub_labels.</p>
</li>
<li>
<p><code>description</code>: This describes the inputs. For instance, <code>stmt=torch.add(x, y)</code>
can be run over several values of <code>x</code> and <code>y</code>. Each pair should be given its
own <code>description</code>, which allows them to appear in separate columns.
Statistics do not mix values of different descriptions, since comparing the
run time of drastically different inputs is generally not meaningful.</p>
</li>
<li>
<p><code>env</code>: An optional description of the torch environment. (e.g. <code>master</code> or
<code>my_branch</code>). Like sub_labels, statistics are calculated across envs. (Since
comparing a branch to master or a stable release is a common use case.)
However <code>Compare</code> will visually group rows which are run with the same <code>env</code>.</p>
</li>
<li>
<p><code>num_threads</code>: By default, <code>Timer</code> will run in single-threaded mode. If
<code>Measurements</code> with different numbers of threads are given to <code>Compare</code>, they
will be grouped into separate blocks of rows.</p>
</li>
</ul>
<h2>Fuzzing</h2>
<p>The <code>Fuzzer</code> class is designed to allow very flexible and repeatable
construction of a wide variety of Tensors while automating away some
of the tedium that comes with creating good benchmark inputs. The two
APIs of interest are the constructor and <code>Fuzzer.take(self, n: int)</code>.
At construction, a <code>Fuzzer</code> is a spec for the kind of Tensors that
should be created. It takes a list of <code>FuzzedParameters</code>, a list of
<code>FuzzedTensors</code>, and an integer with which to seed the Fuzzer.</p>
<p>The reason for distinguishing between parameters and Tensors is that the shapes
and data of Tensors is often linked (e.g. shapes must be identical or
broadcastable, indices must fall within a given range, etc.) As a result we
must first materialize values for each parameter, and then use them to
construct Tensors in a second pass. As a concrete reference, the following
will create Tensors <code>x</code> and <code>y</code>, where <code>x</code> is a 2D Tensor and <code>y</code> is
broadcastable to the shape of <code>x</code>:</p>
<p><code>fuzzer = Fuzzer(
  parameters=[
    FuzzedParameter("k0", 16, 16 * 1024, "loguniform"),
    FuzzedParameter("k1", 16, 16 * 1024, "loguniform"),
  ],
  tensors=[
    FuzzedTensor(
      name="x", size=("k0", "k1"), probability_contiguous=0.75
    ),
    FuzzedTensor(
      name="y", size=("k0", 1), probability_contiguous=0.75
    ),
  ],
  seed=0,
)</code></p>
<p>Calling <code>fuzzer.take(n)</code> will create a generator with <code>n</code> elements which
yields randomly generated Tensors satisfying the above definition, as well
as some metadata about the parameters and Tensors. Critically, calling
<code>.take(...)</code> multiple times will produce generators which select the same
parameters, allowing repeat measurements and different environments to
conduct the same trial. <code>FuzzedParameter</code> and <code>FuzzedTensor</code> support a
fairly involved set of behaviors to reflect the rich character of Tensor
operations and representations. (For instance, note the
<code>probability_contiguous</code> argument which signals that some fraction of the
time non-contiguous Tensors should be created.) The best way to understand
<code>Fuzzer</code>, however, is probably to experiment with <code>examples.fuzzer</code>.</p>
<h1>Examples:</h1>
<p><code>python -m examples.simple_timeit</code></p>
<p><code>python -m examples.compare</code></p>
<p><code>python -m examples.fuzzer</code></p>
<p><code>python -m examples.end_to_end</code></p>