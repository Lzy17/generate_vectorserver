<h1>GraphDef Regularization</h1>
<p>This directory contains the code for TensorFlow GraphDef regularization,
sometimes referred to as "canonicalization".</p>
<h2>What does it mean to "regularize" a GraphDef?</h2>
<p>The TensorFlow GraphDef is the representation of TensorFlow programs. It shares
a lot of the characteristics of an
<a href="https://en.wikipedia.org/wiki/Intermediate_representation">intermediate representation</a>
or IR. A single TensorFlow program can produce different GraphDefs, depending on
the device, platform, TF version, runtime state, etc.</p>
<p>"Regularization" is the process of removing this non-determinism from the
GraphDef.</p>
<h2>Interesting Problems</h2>
<p>GraphDef regularization helps us answer a variety of interesting questions:</p>
<ul>
<li><a href="https://en.wikipedia.org/wiki/Graph_isomorphism">Graph Isomorphism</a>: Do two
  GraphDefs describe the same program?</li>
<li><a href="https://github.com/tensorflow/community/pull/415">Graph Fingerprinting</a>: How
  can we can uniquely identify a graph using a much shorter fingerprint?</li>
</ul>
<h2>Algorithms</h2>
<ul>
<li><code>simple_delete</code>: An algorithm that deletes parts of the GraphDef that are not
   deterministic.</li>
</ul>
<h2>Testing</h2>
<ul>
<li>TODO(b/239046865): Complete this section.</li>
</ul>
<h2>Contributions Welcome</h2>
<p>If you would like to contribute to the GraphDef regularization library, please
send us a pull request. We welcome collaboration!</p>