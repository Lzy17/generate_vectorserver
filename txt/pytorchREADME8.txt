<h1>PyTorch Release Scripts</h1>
<p>These are a collection of scripts that are to be used for release activities.</p>
<blockquote>
<p>NOTE: All scripts should do no actual work unless the <code>DRY_RUN</code> environment variable is set
      to <code>disabled</code>.
      The basic idea being that there should be no potential to do anything dangerous unless
      <code>DRY_RUN</code> is explicitly set to <code>disabled</code>.</p>
</blockquote>
<h2>Requirements to actually run these scripts</h2>
<ul>
<li>AWS access to pytorch account</li>
<li>Access to upload conda packages to the <code>pytorch</code> conda channel</li>
<li>Access to the PyPI repositories</li>
</ul>
<h2>Promote</h2>
<p>These are scripts related to promotion of release candidates to GA channels, these
can actually be used to promote pytorch, libtorch, and related domain libraries.</p>
<h3>Usage</h3>
<p>Usage should be fairly straightforward and should actually require no extra variables
if you are running from the correct git tags. (i.e. the GA tag to promote is currently
checked out)</p>
<p><code>PACKAGE_TYPE</code> and <code>PACKAGE_NAME</code> can be swapped out to promote other packages.</p>
<h4>Promoting pytorch wheels</h4>
<p><code>bash
promote/s3_to_s3.sh</code></p>
<h4>Promoting libtorch archives</h4>
<p><code>bash
PACKAGE_TYPE=libtorch PACKAGE_NAME=libtorch promote/s3_to_s3.sh</code></p>
<h4>Promoting conda packages</h4>
<p><code>bash
promote/conda_to_conda.sh</code></p>
<h4>Promoting wheels to PyPI</h4>
<p><strong>WARNING</strong>: These can only be run once and cannot be undone, run with caution
<code>promote/wheel_to_pypi.sh</code></p>
<h2>Restoring backups</h2>
<p>All release candidates are currently backed up to <code>s3://pytorch-backup/${TAG_NAME}</code> and
can be restored to the test channels with the <code>restore-backup.sh</code> script.</p>
<p>Which backup to restore from is dictated by the <code>RESTORE_FROM</code> environment variable.</p>
<h3>Usage</h3>
<p><code>bash
RESTORE_FROM=v1.5.0-rc5 ./restore-backup.sh</code></p>