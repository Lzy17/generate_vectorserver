<h1>pytorch/.github</h1>
<blockquote>
<p>NOTE: This README contains information for the <code>.github</code> directory but cannot be located there because it will overwrite the
repo README.</p>
</blockquote>
<p>This directory contains workflows and scripts to support our CI infrastructure that runs on GitHub Actions.</p>
<h2>Workflows</h2>
<ul>
<li>Pull CI (<code>pull.yml</code>) is run on PRs and on main.</li>
<li>Trunk CI (<code>trunk.yml</code>) is run on trunk to validate incoming commits. Trunk jobs are usually more expensive to run so we do not run them on PRs unless specified.</li>
<li>Scheduled CI (<code>periodic.yml</code>) is a subset of trunk CI that is run every few hours on main.</li>
<li>Binary CI is run to package binaries for distribution for all platforms.</li>
</ul>
<h2>Templates</h2>
<p>Templates written in <a href="https://jinja.palletsprojects.com/en/3.0.x/">Jinja</a> are located in the <code>.github/templates</code> directory
and used to generate workflow files for binary jobs found in the <code>.github/workflows/</code> directory. These are also a
couple of utility templates used to discern common utilities that can be used amongst different templates.</p>
<h3>(Re)Generating workflow files</h3>
<p>You will need <code>jinja2</code> in order to regenerate the workflow files which can be installed using:
<code>bash
pip install -r .github/requirements/regenerate-requirements.txt</code></p>
<p>Workflows can be generated / regenerated using the following command:
<code>bash
.github/regenerate.sh</code></p>
<h3>Adding a new generated binary workflow</h3>
<p>New generated binary workflows can be added in the <code>.github/scripts/generate_ci_workflows.py</code> script. You can reference
examples from that script in order to add the workflow to the stream that is relevant to what you particularly
care about.</p>
<p>Different parameters can be used to achieve different goals, i.e. running jobs on a cron, running only on trunk, etc.</p>
<h4>ciflow (trunk)</h4>
<p>The label <code>ciflow/trunk</code> can be used to run <code>trunk</code> only workflows. This is especially useful if trying to re-land a PR that was
reverted for failing a <code>non-default</code> workflow.</p>
<h2>Infra</h2>
<p>Currently most of our self hosted runners are hosted on AWS, for a comprehensive list of available runner types you
can reference <code>.github/scale-config.yml</code>.</p>
<p>Exceptions to AWS for self hosted:
* ROCM runners</p>
<h3>Adding new runner types</h3>
<p>New runner types can be added by committing changes to <code>.github/scale-config.yml</code>. Example: https://github.com/pytorch/pytorch/pull/70474</p>
<blockquote>
<p>NOTE: New runner types can only be used once the changes to <code>.github/scale-config.yml</code> have made their way into the default branch</p>
</blockquote>
<h3>Testing <a href="https://github.com/pytorch/builder">pytorch/builder</a> changes</h3>
<p>In order to test changes to the builder scripts:</p>
<ol>
<li>Specify your builder PR's branch and repo as <code>builder_repo</code> and  <code>builder_branch</code> in <a href="https://github.com/pytorch/pytorch/blob/32356aaee6a77e0ae424435a7e9da3d99e7a4ca5/.github/templates/common.yml.j2#LL10C26-L10C32"><code>.github/templates/common.yml.j2</code></a>.</li>
<li>Regenerate workflow files with <code>.github/regenerate.sh</code> (see above).</li>
<li>Submit fake PR to PyTorch. If changing binaries build, add an appropriate label like <code>ciflow/binaries</code> to trigger the builds.</li>
</ol>