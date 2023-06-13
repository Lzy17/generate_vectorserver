<h1>TensorFlow Python API Upgrade Utility</h1>
<p>This tool allows you to upgrade your existing TensorFlow Python scripts,
specifically:
* <code>tf_upgrade_v2.py</code>: Upgrade code from TensorFlow 1.x to TensorFlow 2.0 preview.
* <code>tf_upgrade.py</code>: Upgrade code to TensorFlow 1.0 from TensorFlow 0.11.</p>
<h2>Running the script from pip package</h2>
<p>First, install TensorFlow pip package*. See
https://www.tensorflow.org/install/pip.</p>
<p>Upgrade script can be run on a single Python file:</p>
<p><code>tf_upgrade_v2 --infile foo.py --outfile foo-upgraded.py</code></p>
<p>It will print a list of errors it finds that it can't fix. You can also run
it on a directory tree:</p>
<p>```</p>
<h1>upgrade the .py files and copy all the other files to the outtree</h1>
<p>tf_upgrade_v2 --intree coolcode --outtree coolcode-upgraded</p>
<h1>just upgrade the .py files</h1>
<p>tf_upgrade_v2 --intree coolcode --outtree coolcode-upgraded --copyotherfiles False
```</p>
<p>*Note: <code>tf_upgrade_v2</code> is installed automatically as a script by the pip install
 after TensorFlow 1.12.</p>
<h2>Report</h2>
<p>The script will also dump out a report e.g. which will detail changes
e.g.:</p>
<p>```
'tensorflow/tools/compatibility/testdata/test_file_v1_12.py' Line 65</p>
<hr />
<p>Added keyword 'input' to reordered function 'tf.argmax'
Renamed keyword argument from 'dimension' to 'axis'</p>
<pre><code>Old:         tf.argmax([[1, 3, 2]], dimension=0)
                                    ~~~~~~~~~~
New:         tf.argmax(input=[[1, 3, 2]], axis=0)
</code></pre>
<p>```</p>
<h2>Caveats</h2>
<ul>
<li>
<p>Don't update parts of your code manually before running this script. In
particular, functions that have had reordered arguments like <code>tf.argmax</code>
or <code>tf.batch_to_space</code> will cause the script to incorrectly add keyword
arguments that mismap arguments.</p>
</li>
<li>
<p>This script wouldn't actually reorder arguments. Instead, the script will add
keyword arguments to functions that had their arguments reordered.</p>
</li>
<li>
<p>The script assumes that <code>tensorflow</code> is imported using <code>import tensorflow as tf</code>.</p>
</li>
<li>
<p>Note for upgrading to 2.0: Check out <a href="http://tf2up.ml">tf2up.ml</a> for a
  convenient tool to upgrade Jupyter notebooks and Python files in a GitHub
  repository.</p>
</li>
<li>
<p>Note for upgrading to 1.0: There are some syntaxes that are not handleable with this script as this
script was designed to use only standard python packages.
If the script fails with "A necessary keyword argument failed to be inserted." or
"Failed to find keyword lexicographically. Fix manually.", you can try
<a href="https://github.com/machrisaa/tf0to1">@machrisaa's fork of this script</a>.
<a href="https://github.com/machrisaa">@machrisaa</a> has used the
<a href="https://redbaron.readthedocs.io/en/latest/">RedBaron Python refactoring engine</a>
which is able to localize syntactic elements more reliably than the built-in
<code>ast</code> module this script is based upon. Note that the alternative script is not
available for TensorFlow 2.0 upgrade.</p>
</li>
</ul>