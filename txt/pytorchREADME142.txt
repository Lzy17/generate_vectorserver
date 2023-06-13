<h1>Warning</h1>
<p>Contents may be out of date. Our CircleCI workflows are gradually being migrated to Github actions.</p>
<h1>Structure of CI</h1>
<p>setup job:
1. Does a git checkout
2. Persists CircleCI scripts (everything in <code>.circleci</code>) into a workspace.  Why?
   We don't always do a Git checkout on all subjobs, but we usually
   still want to be able to call scripts one way or another in a subjob.
   Persisting files this way lets us have access to them without doing a
   checkout.  This workspace is conventionally mounted on <code>~/workspace</code>
   (this is distinguished from <code>~/project</code>, which is the conventional
   working directory that CircleCI will default to starting your jobs
   in.)
3. Write out the commit message to <code>.circleci/COMMIT_MSG</code>.  This is so
   we can determine in subjobs if we should actually run the jobs or
   not, even if there isn't a Git checkout.</p>
<h1>CircleCI configuration generator</h1>
<p>One may no longer make changes to the <code>.circleci/config.yml</code> file directly.
Instead, one must edit these Python scripts or files in the <code>verbatim-sources/</code> directory.</p>
<h2>Usage</h2>
<ol>
<li>Make changes to these scripts.</li>
<li>Run the <code>regenerate.sh</code> script in this directory and commit the script changes and the resulting change to <code>config.yml</code>.</li>
</ol>
<p>You'll see a build failure on GitHub if the scripts don't agree with the checked-in version.</p>
<h2>Motivation</h2>
<p>These scripts establish a single, authoritative source of documentation for the CircleCI configuration matrix.
The documentation, in the form of diagrams, is automatically generated and cannot drift out of sync with the YAML content.</p>
<p>Furthermore, consistency is enforced within the YAML config itself, by using a single source of data to generate
multiple parts of the file.</p>
<ul>
<li>Facilitates one-off culling/enabling of CI configs for testing PRs on special targets</li>
</ul>
<p>Also see https://github.com/pytorch/pytorch/issues/17038</p>
<h2>Future direction</h2>
<h3>Declaring sparse config subsets</h3>
<p>See comment <a href="https://github.com/pytorch/pytorch/pull/17323#pullrequestreview-206945747">here</a>:</p>
<p>In contrast with a full recursive tree traversal of configuration dimensions,</p>
<blockquote>
<p>in the future I think we actually want to decrease our matrix somewhat and have only a few mostly-orthogonal builds that taste as many different features as possible on PRs, plus a more complete suite on every PR and maybe an almost full suite nightly/weekly (we don't have this yet). Specifying PR jobs in the future might be easier to read with an explicit list when we come to this.</p>
</blockquote>
<hr />
<hr />
<h1>How do the binaries / nightlies / releases work?</h1>
<h3>What is a binary?</h3>
<p>A binary or package (used interchangeably) is a pre-built collection of c++ libraries, header files, python bits, and other files. We build these and distribute them so that users do not need to install from source.</p>
<p>A <strong>binary configuration</strong> is a collection of</p>
<ul>
<li>release or nightly<ul>
<li>releases are stable, nightlies are beta and built every night</li>
</ul>
</li>
<li>python version<ul>
<li>linux: 3.7m (mu is wide unicode or something like that. It usually doesn't matter but you should know that it exists)</li>
<li>macos: 3.7, 3.8</li>
<li>windows: 3.7, 3.8</li>
</ul>
</li>
<li>cpu version<ul>
<li>cpu, cuda 9.0, cuda 10.0</li>
<li>The supported cuda versions occasionally change</li>
</ul>
</li>
<li>operating system<ul>
<li>Linux - these are all built on CentOS. There haven't been any problems in the past building on CentOS and using on Ubuntu</li>
<li>MacOS</li>
<li>Windows - these are built on Azure pipelines</li>
</ul>
</li>
<li>devtoolset version (gcc compiler version)<ul>
<li>This only matters on Linux cause only Linux uses gcc. tldr is gcc made a backwards incompatible change from gcc 4.8 to gcc 5, because it had to change how it implemented std::vector and std::string</li>
</ul>
</li>
</ul>
<h3>Where are the binaries?</h3>
<p>The binaries are built in CircleCI. There are nightly binaries built every night at 9pm PST (midnight EST) and release binaries corresponding to Pytorch releases, usually every few months.</p>
<p>We have 3 types of binary packages</p>
<ul>
<li>pip packages - nightlies are stored on s3 (pip install -f \&lt;a s3 url>). releases are stored in a pip repo (pip install torch) (ask Soumith about this)</li>
<li>conda packages - nightlies and releases are both stored in a conda repo. Nighty packages have a '_nightly' suffix</li>
<li>libtorch packages - these are zips of all the c++ libraries, header files, and sometimes dependencies. These are c++ only<ul>
<li>shared with dependencies (the only supported option for Windows)</li>
<li>static with dependencies</li>
<li>shared without dependencies</li>
<li>static without dependencies</li>
</ul>
</li>
</ul>
<p>All binaries are built in CircleCI workflows except Windows. There are checked-in workflows (committed into the .circleci/config.yml) to build the nightlies every night. Releases are built by manually pushing a PR that builds the suite of release binaries (overwrite the config.yml to build the release)</p>
<h1>CircleCI structure of the binaries</h1>
<p>Some quick vocab:</p>
<ul>
<li>A *<em>workflow</em><em> is a CircleCI concept; it is a DAG of '</em><em>jobs</em>*'. ctrl-f 'workflows' on https://github.com/pytorch/pytorch/blob/main/.circleci/config.yml to see the workflows.</li>
<li><strong>jobs</strong> are a sequence of '<strong>steps</strong>'</li>
<li><strong>steps</strong> are usually just a bash script or a builtin CircleCI command. <em>All steps run in new environments, environment variables declared in one script DO NOT persist to following steps</em></li>
<li>CircleCI has a <strong>workspace</strong>, which is essentially a cache between steps of the <em>same job</em> in which you can store artifacts between steps.</li>
</ul>
<h2>How are the workflows structured?</h2>
<p>The nightly binaries have 3 workflows. We have one job (actually 3 jobs:  build, test, and upload) per binary configuration</p>
<ol>
<li>binary_builds<ol>
<li>every day midnight EST</li>
<li>linux: https://github.com/pytorch/pytorch/blob/main/.circleci/verbatim-sources/linux-binary-build-defaults.yml</li>
<li>macos: https://github.com/pytorch/pytorch/blob/main/.circleci/verbatim-sources/macos-binary-build-defaults.yml</li>
<li>For each binary configuration, e.g. linux_conda_3.7_cpu there is a<ol>
<li>binary_linux_conda_3.7_cpu_build<ol>
<li>Builds the build. On linux jobs this uses the 'docker executor'.</li>
<li>Persists the package to the workspace</li>
</ol>
</li>
<li>binary_linux_conda_3.7_cpu_test<ol>
<li>Loads the package to the workspace</li>
<li>Spins up a docker image (on Linux), mapping the package and code repos into the docker</li>
<li>Runs some smoke tests in the docker</li>
<li>(Actually, for macos this is a step rather than a separate job)</li>
</ol>
</li>
<li>binary_linux_conda_3.7_cpu_upload<ol>
<li>Logs in to aws/conda</li>
<li>Uploads the package</li>
</ol>
</li>
</ol>
</li>
</ol>
</li>
<li>update_s3_htmls<ol>
<li>every day 5am EST</li>
<li>https://github.com/pytorch/pytorch/blob/main/.circleci/verbatim-sources/binary_update_htmls.yml</li>
<li>See below for what these are for and why they're needed</li>
<li>Three jobs that each examine the current contents of aws and the conda repo and update some html files in s3</li>
</ol>
</li>
<li>binarysmoketests<ol>
<li>every day</li>
<li>https://github.com/pytorch/pytorch/blob/main/.circleci/verbatim-sources/nightly-build-smoke-tests-defaults.yml</li>
<li>For each binary configuration, e.g. linux_conda_3.7_cpu there is a<ol>
<li>smoke_linux_conda_3.7_cpu<ol>
<li>Downloads the package from the cloud, e.g. using the official pip or conda instructions</li>
<li>Runs the smoke tests</li>
</ol>
</li>
</ol>
</li>
</ol>
</li>
</ol>
<h2>How are the jobs structured?</h2>
<p>The jobs are in https://github.com/pytorch/pytorch/tree/main/.circleci/verbatim-sources. Jobs are made of multiple steps. There are some shared steps used by all the binaries/smokes. Steps of these jobs are all delegated to scripts in https://github.com/pytorch/pytorch/tree/main/.circleci/scripts .</p>
<ul>
<li>Linux jobs: https://github.com/pytorch/pytorch/blob/main/.circleci/verbatim-sources/linux-binary-build-defaults.yml<ul>
<li>binary_linux_build.sh</li>
<li>binary_linux_test.sh</li>
<li>binary_linux_upload.sh</li>
</ul>
</li>
<li>MacOS jobs: https://github.com/pytorch/pytorch/blob/main/.circleci/verbatim-sources/macos-binary-build-defaults.yml<ul>
<li>binary_macos_build.sh</li>
<li>binary_macos_test.sh</li>
<li>binary_macos_upload.sh</li>
</ul>
</li>
<li>Update html jobs: https://github.com/pytorch/pytorch/blob/main/.circleci/verbatim-sources/binary_update_htmls.yml<ul>
<li>These delegate from the pytorch/builder repo</li>
<li>https://github.com/pytorch/builder/blob/main/cron/update_s3_htmls.sh</li>
<li>https://github.com/pytorch/builder/blob/main/cron/upload_binary_sizes.sh</li>
</ul>
</li>
<li>Smoke jobs (both linux and macos): https://github.com/pytorch/pytorch/blob/main/.circleci/verbatim-sources/nightly-build-smoke-tests-defaults.yml<ul>
<li>These delegate from the pytorch/builder repo</li>
<li>https://github.com/pytorch/builder/blob/main/run_tests.sh</li>
<li>https://github.com/pytorch/builder/blob/main/smoke_test.sh</li>
<li>https://github.com/pytorch/builder/blob/main/check_binary.sh</li>
</ul>
</li>
<li>Common shared code (shared across linux and macos): https://github.com/pytorch/pytorch/blob/main/.circleci/verbatim-sources/nightly-binary-build-defaults.yml<ul>
<li>binary_checkout.sh - checks out pytorch/builder repo. Right now this also checks out pytorch/pytorch, but it shouldn't. pytorch/pytorch should just be shared through the workspace. This can handle being run before binary_populate_env.sh</li>
<li>binary_populate_env.sh - parses BUILD_ENVIRONMENT into the separate env variables that make up a binary configuration. Also sets lots of default values, the date, the version strings, the location of folders in s3, all sorts of things. This generally has to be run before other steps.</li>
<li>binary_install_miniconda.sh - Installs miniconda, cross platform. Also hacks this for the update_binary_sizes job that doesn't have the right env variables</li>
<li>binary_run_in_docker.sh - Takes a bash script file (the actual test code) from a hardcoded location, spins up a docker image, and runs the script inside the docker image</li>
</ul>
</li>
</ul>
<h3><strong>Why do the steps all refer to scripts?</strong></h3>
<p>CircleCI creates a  final yaml file by inlining every &lt;&lt;* segment, so if we were to keep all the code in the config.yml itself then the config size would go over 4 MB and cause infra problems.</p>
<h3><strong>What is binary_run_in_docker for?</strong></h3>
<p>So, CircleCI has several executor types: macos, machine, and docker are the ones we use. The 'machine' executor gives you two cores on some linux vm. The 'docker' executor gives you considerably more cores (nproc was 32 instead of 2 back when I tried in February). Since the dockers are faster, we try to run everything that we can in dockers. Thus</p>
<ul>
<li>linux build jobs use the docker executor. Running them on the docker executor was at least 2x faster than running them on the machine executor</li>
<li>linux test jobs use the machine executor in order for them to properly interface with GPUs since docker executors cannot execute with attached GPUs</li>
<li>linux upload jobs use the machine executor. The upload jobs are so short that it doesn't really matter what they use</li>
<li>linux smoke test jobs use the machine executor for the same reason as the linux test jobs</li>
</ul>
<p>binary_run_in_docker.sh is a way to share the docker start-up code between the binary test jobs and the binary smoke test jobs</p>
<h3><strong>Why does binary_checkout also checkout pytorch? Why shouldn't it?</strong></h3>
<p>We want all the nightly binary jobs to run on the exact same git commit, so we wrote our own checkout logic to ensure that the same commit was always picked. Later circleci changed that to use a single pytorch checkout and persist it through the workspace (they did this because our config file was too big, so they wanted to take a lot of the setup code into scripts, but the scripts needed the code repo to exist to be called, so they added a prereq step called 'setup' to checkout the code and persist the needed scripts to the workspace). The changes to the binary jobs were not properly tested, so they all broke from missing pytorch code no longer existing. We hotfixed the problem by adding the pytorch checkout back to binary_checkout, so now there's two checkouts of pytorch on the binary jobs. This problem still needs to be fixed, but it takes careful tracing of which code is being called where.</p>
<h1>Code structure of the binaries (circleci agnostic)</h1>
<h2>Overview</h2>
<p>The code that runs the binaries lives in two places, in the normal <a href="http://github.com/pytorch/pytorch">github.com/pytorch/pytorch</a>, but also in <a href="http://github.com/pytorch/builder">github.com/pytorch/builder</a>, which is a repo that defines how all the binaries are built. The relevant code is</p>
<p>```</p>
<h1>All code needed to set-up environments for build code to run in,</h1>
<h1>but only code that is specific to the current CI system</h1>
<p>pytorch/pytorch
- .circleci/                # Folder that holds all circleci related stuff
  - config.yml              # GENERATED file that actually controls all circleci behavior
  - verbatim-sources        # Used to generate job/workflow sections in ^
  - scripts/                # Code needed to prepare circleci environments for binary build scripts
- setup.py                  # Builds pytorch. This is wrapped in pytorch/builder
- cmake files               # used in normal building of pytorch</p>
<h1>All code needed to prepare a binary build, given an environment</h1>
<h1>with all the right variables/packages/paths.</h1>
<p>pytorch/builder</p>
<h1>Given an installed binary and a proper python env, runs some checks</h1>
<h1>to make sure the binary was built the proper way. Checks things like</h1>
<h1>the library dependencies, symbols present, etc.</h1>
<ul>
<li>check_binary.sh</li>
</ul>
<h1>Given an installed binary, runs python tests to make sure everything</h1>
<h1>is in order. These should be de-duped. Right now they both run smoke</h1>
<h1>tests, but are called from different places. Usually just call some</h1>
<h1>import statements, but also has overlap with check_binary.sh above</h1>
<ul>
<li>run_tests.sh</li>
<li>smoke_test.sh</li>
</ul>
<h1>Folders that govern how packages are built. See paragraphs below</h1>
<ul>
<li>conda/</li>
<li>build_pytorch.sh          # Entrypoint. Delegates to proper conda build folder</li>
<li>switch_cuda_version.sh    # Switches activate CUDA installation in Docker</li>
<li>pytorch-nightly/          # Build-folder</li>
<li>manywheel/</li>
<li>build_cpu.sh              # Entrypoint for cpu builds</li>
<li>build.sh                  # Entrypoint for CUDA builds</li>
<li>build_common.sh           # Actual build script that ^^ call into</li>
<li>wheel/</li>
<li>build_wheel.sh            # Entrypoint for wheel builds</li>
<li>windows/</li>
<li>build_pytorch.bat         # Entrypoint for wheel builds on Windows
```</li>
</ul>
<p>Every type of package has an entrypoint build script that handles the all the important logic.</p>
<h2>Conda</h2>
<p>Linux, MacOS and Windows use the same code flow for the conda builds.</p>
<p>Conda packages are built with conda-build, see https://conda.io/projects/conda-build/en/latest/resources/commands/conda-build.html</p>
<p>Basically, you pass <code>conda build</code> a build folder (pytorch-nightly/ above) that contains a build script and a meta.yaml. The meta.yaml specifies in what python environment to build the package in, and what dependencies the resulting package should have, and the build script gets called in the env to build the thing.
tl;dr on conda-build is</p>
<ol>
<li>Creates a brand new conda environment, based off of deps in the meta.yaml<ol>
<li>Note that environment variables do not get passed into this build env unless they are specified in the meta.yaml</li>
<li>If the build fails this environment will stick around. You can activate it for much easier debugging. The “General Python” section below explains what exactly a python “environment” is.</li>
</ol>
</li>
<li>Calls build.sh in the environment</li>
<li>Copies the finished package to a new conda env, also specified by the meta.yaml</li>
<li>Runs some simple import tests (if specified in the meta.yaml)</li>
<li>Saves the finished package as a tarball</li>
</ol>
<p>The build.sh we use is essentially a wrapper around <code>python setup.py build</code>, but it also manually copies in some of our dependent libraries into the resulting tarball and messes with some rpaths.</p>
<p>The entrypoint file <code>builder/conda/build_conda.sh</code> is complicated because</p>
<ul>
<li>It works for Linux, MacOS and Windows<ul>
<li>The mac builds used to create their own environments, since they all used to be on the same machine. There’s now a lot of extra logic to handle conda envs. This extra machinery could be removed</li>
</ul>
</li>
<li>It used to handle testing too, which adds more logic messing with python environments too. This extra machinery could be removed.</li>
</ul>
<h2>Manywheels (linux pip and libtorch packages)</h2>
<p>Manywheels are pip packages for linux distros. Note that these manywheels are not actually manylinux compliant.</p>
<p><code>builder/manywheel/build_cpu.sh</code> and <code>builder/manywheel/build.sh</code> (for CUDA builds) just set different env vars and then call into <code>builder/manywheel/build_common.sh</code></p>
<p>The entrypoint file <code>builder/manywheel/build_common.sh</code> is really really complicated because</p>
<ul>
<li>This used to handle building for several different python versions at the same time. The loops have been removed, but there's still unnecessary folders and movements here and there.<ul>
<li>The script is never used this way anymore. This extra machinery could be removed.</li>
</ul>
</li>
<li>This used to handle testing the pip packages too. This is why there’s testing code at the end that messes with python installations and stuff<ul>
<li>The script is never used this way anymore. This extra machinery could be removed.</li>
</ul>
</li>
<li>This also builds libtorch packages<ul>
<li>This should really be separate. libtorch packages are c++ only and have no python. They should not share infra with all the python specific stuff in this file.</li>
</ul>
</li>
<li>There is a lot of messing with rpaths. This is necessary, but could be made much much simpler if the above issues were fixed.</li>
</ul>
<h2>Wheels (MacOS pip and libtorch packages)</h2>
<p>The entrypoint file <code>builder/wheel/build_wheel.sh</code> is complicated because</p>
<ul>
<li>The mac builds used to all run on one machine (we didn’t have autoscaling mac machines till circleci). So this script handled siloing itself by setting-up and tearing-down its build env and siloing itself into its own build directory.<ul>
<li>The script is never used this way anymore. This extra machinery could be removed.</li>
</ul>
</li>
<li>This also builds libtorch packages<ul>
<li>Ditto the comment above. This should definitely be separated out.</li>
</ul>
</li>
</ul>
<p>Note that the MacOS Python wheels are still built in conda environments. Some of the dependencies present during build also come from conda.</p>
<h2>Windows Wheels (Windows pip and libtorch packages)</h2>
<p>The entrypoint file <code>builder/windows/build_pytorch.bat</code> is complicated because</p>
<ul>
<li>This used to handle building for several different python versions at the same time. This is why there are loops everywhere<ul>
<li>The script is never used this way anymore. This extra machinery could be removed.</li>
</ul>
</li>
<li>This used to handle testing the pip packages too. This is why there’s testing code at the end that messes with python installations and stuff<ul>
<li>The script is never used this way anymore. This extra machinery could be removed.</li>
</ul>
</li>
<li>This also builds libtorch packages<ul>
<li>This should really be separate. libtorch packages are c++ only and have no python. They should not share infra with all the python specific stuff in this file.</li>
</ul>
</li>
</ul>
<p>Note that the Windows Python wheels are still built in conda environments. Some of the dependencies present during build also come from conda.</p>
<h2>General notes</h2>
<h3>Note on run_tests.sh, smoke_test.sh, and check_binary.sh</h3>
<ul>
<li>These should all be consolidated</li>
<li>These must run on all OS types: MacOS, Linux, and Windows</li>
<li>These all run smoke tests at the moment. They inspect the packages some, maybe run a few import statements. They DO NOT run the python tests nor the cpp tests. The idea is that python tests on main and PR merges will catch all breakages. All these tests have to do is make sure the special binary machinery didn’t mess anything up.</li>
<li>There are separate run_tests.sh and smoke_test.sh because one used to be called by the smoke jobs and one used to be called by the binary test jobs (see circleci structure section above). This is still true actually, but these could be united into a single script that runs these checks, given an installed pytorch package.</li>
</ul>
<h3>Note on libtorch</h3>
<p>Libtorch packages are built in the wheel build scripts: manywheel/build_*.sh for linux and build_wheel.sh for mac. There are several things wrong with this</p>
<ul>
<li>It’s confusing. Most of those scripts deal with python specifics.</li>
<li>The extra conditionals everywhere severely complicate the wheel build scripts</li>
<li>The process for building libtorch is different from the official instructions (a plain call to cmake, or a call to a script)</li>
</ul>
<h3>Note on docker images / Dockerfiles</h3>
<p>All linux builds occur in docker images. The docker images are</p>
<ul>
<li>pytorch/conda-cuda<ul>
<li>Has ALL CUDA versions installed. The script pytorch/builder/conda/switch_cuda_version.sh sets /usr/local/cuda to a symlink to e.g. /usr/local/cuda-10.0 to enable different CUDA builds</li>
<li>Also used for cpu builds</li>
</ul>
</li>
<li>pytorch/manylinux-cuda90</li>
<li>pytorch/manylinux-cuda100<ul>
<li>Also used for cpu builds</li>
</ul>
</li>
</ul>
<p>The Dockerfiles are available in pytorch/builder, but there is no circleci job or script to build these docker images, and they cannot be run locally (unless you have the correct local packages/paths). Only Soumith can build them right now.</p>
<h3>General Python</h3>
<ul>
<li>This is still a good explanation of python installations https://caffe2.ai/docs/faq.html#why-do-i-get-import-errors-in-python-when-i-try-to-use-caffe2</li>
</ul>
<h1>How to manually rebuild the binaries</h1>
<p>tl;dr make a PR that looks like https://github.com/pytorch/pytorch/pull/21159</p>
<p>Sometimes we want to push a change to mainand then rebuild all of today's binaries after that change. As of May 30, 2019 there isn't a way to manually run a workflow in the UI. You can manually re-run a workflow, but it will use the exact same git commits as the first run and will not include any changes. So we have to make a PR and then force circleci to run the binary workflow instead of the normal tests. The above PR is an example of how to do this; essentially you copy-paste the binarybuilds workflow steps into the default workflow steps. If you need to point the builder repo to a different commit then you'd need to change https://github.com/pytorch/pytorch/blob/main/.circleci/scripts/binary_checkout.sh#L42-L45 to checkout what you want.</p>
<h2>How to test changes to the binaries via .circleci</h2>
<p>Writing PRs that test the binaries is annoying, since the default circleci jobs that run on PRs are not the jobs that you want to run. Likely, changes to the binaries will touch something under .circleci/ and require that .circleci/config.yml be regenerated (.circleci/config.yml controls all .circleci behavior, and is generated using <code>.circleci/regenerate.sh</code> in python 3.7). But you also need to manually hardcode the binary jobs that you want to test into the .circleci/config.yml workflow, so you should actually make at least two commits, one for your changes and one to temporarily hardcode jobs. See https://github.com/pytorch/pytorch/pull/22928 as an example of how to do this.</p>
<p>```sh</p>
<h1>Make your changes</h1>
<p>touch .circleci/verbatim-sources/nightly-binary-build-defaults.yml</p>
<h1>Regenerate the yaml, has to be in python 3.7</h1>
<p>.circleci/regenerate.sh</p>
<h1>Make a commit</h1>
<p>git add .circleci *
git commit -m "My real changes"
git push origin my_branch</p>
<h1>Now hardcode the jobs that you want in the .circleci/config.yml workflows section</h1>
<h1>Also eliminate ensure-consistency and should_run_job checks</h1>
<h1>e.g. https://github.com/pytorch/pytorch/commit/2b3344bfed8772fe86e5210cc4ee915dee42b32d</h1>
<h1>Make a commit you won't keep</h1>
<p>git add .circleci
git commit -m "[DO NOT LAND] testing binaries for above changes"
git push origin my_branch</p>
<h1>Now you need to make some changes to the first commit.</h1>
<p>git rebase -i HEAD~2 # mark the first commit as 'edit'</p>
<h1>Make the changes</h1>
<p>touch .circleci/verbatim-sources/nightly-binary-build-defaults.yml
.circleci/regenerate.sh</p>
<h1>Ammend the commit and recontinue</h1>
<p>git add .circleci
git commit --amend
git rebase --continue</p>
<h1>Update the PR, need to force since the commits are different now</h1>
<p>git push origin my_branch --force
```</p>
<p>The advantage of this flow is that you can make new changes to the base commit and regenerate the .circleci without having to re-write which binary jobs you want to test on. The downside is that all updates will be force pushes.</p>
<h2>How to build a binary locally</h2>
<h3>Linux</h3>
<p>You can build Linux binaries locally easily using docker.</p>
<p>```sh</p>
<h1>Run the docker</h1>
<h1>Use the correct docker image, pytorch/conda-cuda used here as an example</h1>
<h1></h1>
<h1>-v path/to/foo:path/to/bar makes path/to/foo on your local machine (the</h1>
<h1>machine that you're running the command on) accessible to the docker</h1>
<h1>container at path/to/bar. So if you then run <code>touch path/to/bar/baz</code></h1>
<h1>in the docker container then you will see path/to/foo/baz on your local</h1>
<h1>machine. You could also clone the pytorch and builder repos in the docker.</h1>
<h1></h1>
<h1>If you know how, add ccache as a volume too and speed up everything</h1>
<p>docker run \
    -v your/pytorch/repo:/pytorch \
    -v your/builder/repo:/builder \
    -v where/you/want/packages/to/appear:/final_pkgs \
    -it pytorch/conda-cuda /bin/bash</p>
<h1>Export whatever variables are important to you. All variables that you'd</h1>
<h1>possibly need are in .circleci/scripts/binary_populate_env.sh</h1>
<h1>You should probably always export at least these 3 variables</h1>
<p>export PACKAGE_TYPE=conda
export DESIRED_PYTHON=3.7
export DESIRED_CUDA=cpu</p>
<h1>Call the entrypoint</h1>
<h1><code>|&amp; tee foo.log</code> just copies all stdout and stderr output to foo.log</h1>
<h1>The builds generate lots of output so you probably need this when</h1>
<h1>building locally.</h1>
<p>/builder/conda/build_pytorch.sh |&amp; tee build_output.log
```</p>
<p><strong>Building CUDA binaries on docker</strong></p>
<p>You can build CUDA binaries on CPU only machines, but you can only run CUDA binaries on CUDA machines. This means that you can build a CUDA binary on a docker on your laptop if you so choose (though it’s gonna take a long time).</p>
<p>For Facebook employees, ask about beefy machines that have docker support and use those instead of your laptop; it will be 5x as fast.</p>
<h3>MacOS</h3>
<p>There’s no easy way to generate reproducible hermetic MacOS environments. If you have a Mac laptop then you can try emulating the .circleci environments as much as possible, but you probably have packages in /usr/local/, possibly installed by brew, that will probably interfere with the build. If you’re trying to repro an error on a Mac build in .circleci and you can’t seem to repro locally, then my best advice is actually to iterate on .circleci    :/</p>
<p>But if you want to try, then I’d recommend</p>
<p>```sh</p>
<h1>Create a new terminal</h1>
<h1>Clear your LD_LIBRARY_PATH and trim as much out of your PATH as you</h1>
<h1>know how to do</h1>
<h1>Install a new miniconda</h1>
<h1>First remove any other python or conda installation from your PATH</h1>
<h1>Always install miniconda 3, even if building for Python &lt;3</h1>
<p>new_conda="~/my_new_conda"
conda_sh="$new_conda/install_miniconda.sh"
curl -o "$conda_sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x "$conda_sh"
"$conda_sh" -b -p "$MINICONDA_ROOT"
rm -f "$conda_sh"
export PATH="~/my_new_conda/bin:$PATH"</p>
<h1>Create a clean python env</h1>
<h1>All MacOS builds use conda to manage the python env and dependencies</h1>
<h1>that are built with, even the pip packages</h1>
<p>conda create -yn binary python=2.7
conda activate binary</p>
<h1>Export whatever variables are important to you. All variables that you'd</h1>
<h1>possibly need are in .circleci/scripts/binary_populate_env.sh</h1>
<h1>You should probably always export at least these 3 variables</h1>
<p>export PACKAGE_TYPE=conda
export DESIRED_PYTHON=3.7
export DESIRED_CUDA=cpu</p>
<h1>Call the entrypoint you want</h1>
<p>path/to/builder/wheel/build_wheel.sh
```</p>
<p>N.B. installing a brand new miniconda is important. This has to do with how conda installations work. See the “General Python” section above, but tldr; is that</p>
<ol>
<li>You make the ‘conda’ command accessible by prepending <code>path/to/conda_root/bin</code> to your PATH.</li>
<li>You make a new env and activate it, which then also gets prepended to your PATH. Now you have <code>path/to/conda_root/envs/new_env/bin:path/to/conda_root/bin:$PATH</code></li>
<li>Now say you (or some code that you ran) call python executable <code>foo</code><ol>
<li>if you installed <code>foo</code> in <code>new_env</code>, then <code>path/to/conda_root/envs/new_env/bin/foo</code> will get called, as expected.</li>
<li>But if you forgot to installed <code>foo</code> in <code>new_env</code> but happened to previously install it in your root conda env (called ‘base’), then unix/linux will still find <code>path/to/conda_root/bin/foo</code> . This is dangerous, since <code>foo</code> can be a different version than you want; <code>foo</code> can even be for an incompatible python version!</li>
</ol>
</li>
</ol>
<p>Newer conda versions and proper python hygiene can prevent this, but just install a new miniconda to be safe.</p>
<h3>Windows</h3>
<p>TODO: fill in</p>