<h2>functorch docs build</h2>
<h2>Build Locally</h2>
<p>Install requirements:
<code>pip install -r requirements.txt</code></p>
<p>One may also need to install <a href="https://pandoc.org/installing.html">pandoc</a>. On Linux we can use: <code>sudo apt-get install pandoc</code>. Or using <code>conda</code> we can use: <code>conda install -c conda-forge pandoc</code>.</p>
<p>To run the docs build:
<code>make html</code></p>
<p>Check out the output files in <code>build/html</code>.</p>
<h2>Deploy</h2>
<p>The functorch docs website does not updated automatically. We need to periodically regenerate it.</p>
<p>You need write permissions to functorch to do this. We use GitHub Pages to serve docs.</p>
<ol>
<li>Build the docs</li>
<li>Save the build/html folder somewhere</li>
<li>Checkout the branch <code>gh-pages</code>.</li>
<li>Delete the contents of the branch and replace it with the build/html folder. <code>index.html</code> should be at the root.</li>
<li>Commit the changes and push the changes to the <code>gh-pages</code> branch.</li>
</ol>