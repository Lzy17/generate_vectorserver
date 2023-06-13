<h1>Autotag</h1>
<h2>How to use</h2>
<p>The tag script can simply be invoked by passing it as a python script:</p>
<p><code>sh
python3 tag_script.py --help</code></p>
<p>To generate the changelog from 5.0.0 up to and including 5.4.3:</p>
<p><code>sh
python3 tag_script.py -t &lt;GITHUB_TOKEN&gt; --no-release --no-pulls --do-previous --compile_file ../../CHANGELOG.md --branch release/rocm-rel-5.4 5.4.3</code></p>
<blockquote>
<p><strong>Note</strong></p>
<p>Compiling the changelog without the <code>--do-previous</code>-flag will always think that all libraries are new since no previous version of said library has been parsed.</p>
</blockquote>
<p>Trying to run without a token is possible but GitHub enforces stricter rate limits and is therefore not advised.</p>