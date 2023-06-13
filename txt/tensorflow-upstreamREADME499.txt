<h1>AutoGraph reference tests</h1>
<p>This directory contains tests for various Python idioms under AutoGraph.
Since they are easy to read, they also double as small code samples.
For constructs which are not supported, these tests indicate the error being
raised.</p>
<p>The BUILD file contains the full list of tests.</p>
<h2>Locating the samples inside tests</h2>
<p>Each test is structured as:</p>
<pre><code>&lt;imports&gt;

&lt;sample functions&gt;

&lt;test class&gt;
</code></pre>
<p>The sample functions are what demonstrate how code is authored for AutoGraph.</p>
<p>The test in generale ensure that the sample code produces the same results when
run in a TF graph as it would when executed as regular Python.</p>