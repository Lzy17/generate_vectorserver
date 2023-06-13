<h1>MIOpenDriver</h1>
<p>The <code>MIOpenDriver</code> enables the user to test the functionality of any particular 
layer in MIOpen in both the forward and backward direction. MIOpen is shipped with <code>MIOpenDriver</code> and its install directory is <code>miopen/bin</code> located in the install directory path.</p>
<h2>Building the Driver</h2>
<p>MIOpenDriver can be build by typing:</p>
<p><code>make MIOpenDriver</code> from the <code>build</code> directory.</p>
<h2>Base Arguments</h2>
<p>All the supported layers in MIOpen can be found by the supported <code>base_args</code> here:</p>
<p><code>./bin/MIOpenDriver --help</code></p>
<p>The supported base arguments:</p>
<ul>
<li><code>conv</code> - Convolutions</li>
<li><code>CBAInfer</code> - Convolution+Bias+Activation fusions for inference</li>
<li><code>pool</code> - Pooling</li>
<li><code>lrn</code> - Local Response Normalization</li>
<li><code>activ</code> - Activations</li>
<li><code>softmax</code> - Softmax</li>
<li><code>bnorm</code> - Batch Normalization</li>
<li><code>rnn</code> - Recurrent Neural Networks (including LSTM and GRU)</li>
<li><code>gemm</code> - General Matrix Multiplication</li>
<li><code>ctc</code> - CTC Loss Function</li>
</ul>
<p>These base arguments support fp32 float type, but some of the drivers suport further datatypes -- specifically, half precision (fp16), brain float16 (bfp16), and 8-bit integers (int8).
 To toggle half precision simpily add the suffix <code>fp16</code> to end of the base argument; e.g., <code>convfp16</code>.
 Likewise, to toggle brain float16 just add the suffix <code>bfp16</code>, and to use 8-bit integers add <code>int8</code>.</p>
<p>Notes for this release:
  * Only convolutions support bfp16 and int8
  * RNN's support fp16 but only on the HIP backend
  * CTC loss function only supports fp32</p>
<p>Summary of base_args meant for different datatypes and different operations:</p>
<p><img alt="DatatypeSupport" src="../docs/data/driverTableCrop.png" /></p>
<h2>Executing MIOpenDriver</h2>
<p>To execute from the build directory: </p>
<p><code>./bin/MIOpenDriver *base_arg* *layer_specific_args*</code></p>
<p>Or to execute the default configuration simpily run: </p>
<p><code>./bin/MIOpenDriver *base_arg*</code></p>
<p>MIOpenDriver example usages:</p>
<ul>
<li>Convolution with search on:</li>
</ul>
<p><code>./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2</code>   </p>
<ul>
<li>Forward convolution with search off:</li>
</ul>
<p><code>./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1</code>  </p>
<ul>
<li>Convolution with half or bfloat16 input type</li>
</ul>
<p><code>./bin/MIOpenDriver convfp16 -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1</code>
<code>./bin/MIOpenDriver convbfp16 -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1</code></p>
<ul>
<li>Pooling with default parameters:</li>
</ul>
<p><code>./bin/MIOpenDriver pool</code>  </p>
<ul>
<li>LRN with default parameters and timing on:</li>
</ul>
<p><code>./bin/MIOpenDriver lrn -t 1</code></p>
<ul>
<li>Batch normalization with spatial fwd train, saving mean and variance tensors:</li>
</ul>
<p><code>./bin/MIOpenDriver bnorm -F 1 -n 32 -c 512 -H 16 -W 16 -m 1 -s 1</code>  </p>
<ul>
<li>RNN with forward and backwards pass, no bias, bi-directional and LSTM mode</li>
</ul>
<p><code>./bin/MIOpenDriver rnn -n 4,4,4,3,3,3,2,2,2,1 -k 10 -H 512 -W 1024 -l 3 -F 0 -b 0 -r 1 -m lstm</code></p>
<ul>
<li>Printout layer specific input arguments:</li>
</ul>
<p><code>./bin/MIOpenDriver *base_arg* -?</code> <strong>OR</strong>  <code>./bin/MIOpenDriver *base_arg* -h (--help)</code></p>
<p>Note: By default the CPU verification is turned on. Verification can be disabled using <code>-V 0</code>.</p>