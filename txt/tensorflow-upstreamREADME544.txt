<h1>TFLite Buffer-Stripping Tool/Library</h1>
<p><strong>NOTE: This is an advanced tool used to reduce bandwidth usage in Neural
Architecture Search applications. Use with caution.</strong></p>
<p>The tools in this directory make it easier to distribute TFLite models to
multiple devices over networks with the sole aim of benchmarking <em>latency</em>
performance. The intended workflow is as follows:</p>
<ul>
<li>The stripping tool empties eligible constants from a TFLite flatbuffer to
    reduce its size.</li>
<li>This lean model can be easily transported to devices over a network.</li>
<li>The reconstitution tool on the device takes in a flatbuffer in memory, and
    fills in the appropriate buffers with random data.</li>
</ul>
<p>As an example, see the before/after sizes for MobileNetV1:</p>
<ul>
<li>Float: 16.9MB -&gt; 12KB</li>
<li>Quantized: 4.3MB -&gt; 17.6 KB</li>
</ul>
<p><strong>NOTE: This tool only supports single subgraphs for now.</strong></p>
<p>There are two tools in this directory:</p>
<h2>1. Stripping buffers out of TFLite flatbuffers</h2>
<p>This tool takes in an input <code>flatbuffer</code>, and strips out (or 'empties') the
buffers (constant data) for tensors that follow the following guidelines:</p>
<ul>
<li>Are either of: Float32, Int32, UInt8, Int8</li>
<li>If Int32, the tensor should have a min of 10 elements</li>
</ul>
<p>The second rule above protects us from invalidating constant data that cannot be
randomised (for example, Reshape 'shape' input).</p>
<p>To run the associated script:</p>
<p><code>bazel run -c opt tensorflow/lite/tools/strip_buffers:strip_buffers_from_fb -- --input_flatbuffer=/input/path.tflite --output_flatbuffer=/output/path.tflite</code></p>
<h2>2. Stripping buffers out of TFLite flatbuffers</h2>
<p>The idea here is to reconstitute the lean flatbuffer <code>Model</code> generared in the
above step, by filling in random data whereever necessary.</p>
<p>The prototype script can be called as:</p>
<p><code>bazel run -c opt tensorflow/lite/tools/strip_buffers:reconstitute_buffers_into_fb -- --input_flatbuffer=/input/path.tflite --output_flatbuffer=/output/path.tflite</code></p>
<h2>C++ Library</h2>
<p>Both the above tools are present as <code>stripping_lib</code> in this directory, which
mutate the flatbuffer(s) in-memory. This ensures we can do the above two steps
without touching the filesystem again.</p>