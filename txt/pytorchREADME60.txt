<h1>PyTorch CI Stats</h1>
<p>We track various stats about each CI job.</p>
<ol>
<li>Jobs upload their artifacts to an intermediate data store (either GitHub
   Actions artifacts or S3, depending on what permissions the job has). Example:
   https://github.com/pytorch/pytorch/blob/a9f6a35a33308f3be2413cc5c866baec5cfe3ba1/.github/workflows/_linux-build.yml#L144-L151</li>
<li>When a workflow completes, a <code>workflow_run</code> event <a href="https://github.com/pytorch/pytorch/blob/d9fca126fca7d7780ae44170d30bda901f4fe35e/.github/workflows/upload-test-stats.yml#L4">triggers
   <code>upload-test-stats.yml</code></a>.</li>
<li><code>upload-test-stats</code> downloads the raw stats from the intermediate data store
   and uploads them as JSON to Rockset, our metrics backend.</li>
</ol>
<p>```mermaid
graph LR
    J1[Job with AWS creds<br>e.g. linux, win] --raw stats--&gt; S3[(AWS S3)]
    J2[Job w/o AWS creds<br>e.g. mac] --raw stats--&gt; GHA[(GH artifacts)]</p>
<pre><code>S3 --&gt; uts[upload-test-stats.yml]
GHA --&gt; uts

uts --json--&gt; R[(Rockset)]
</code></pre>
<p>```</p>
<p>Why this weird indirection? Because writing to Rockset requires special
permissions which, for security reasons, we do not want to give to pull request
CI. Instead, we implemented GitHub's <a href="https://securitylab.github.com/research/github-actions-preventing-pwn-requests/">recommended
pattern</a>
for cases like this.</p>
<p>For more details about what stats we export, check out
<a href="https://github.com/pytorch/pytorch/blob/d9fca126fca7d7780ae44170d30bda901f4fe35e/.github/workflows/upload-test-stats.yml"><code>upload-test-stats.yml</code></a></p>