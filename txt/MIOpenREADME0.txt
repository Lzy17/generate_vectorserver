<h1>MIOpen</h1>
<p>AMD's library for high performance machine learning primitives. 
Sources and binaries can be found at <a href="https://github.com/ROCmSoftwarePlatform/MIOpen">MIOpen's GitHub site</a>.
The latest released documentation can be read online <a href="https://rocmsoftwareplatform.github.io/MIOpen/doc/html/index.html">here</a>.</p>
<p>MIOpen supports two programming models - 
1. <a href="https://github.com/ROCm-Developer-Tools/HIP">HIP</a> (Primary Support).
2. OpenCL.</p>
<h2>Documentation</h2>
<p>For a detailed description of the <strong>MIOpen</strong> library see the <a href="https://rocmdocs.amd.com/projects/MIOpen/en/latest/">Documentation</a>.</p>
<h3>How to build documentation</h3>
<p>Run the steps below to build documentation locally.</p>
<p>```
cd docs</p>
<p>pip3 install -r .sphinx/requirements.txt</p>
<p>python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```</p>
<h2>Prerequisites</h2>
<ul>
<li>More information about ROCm stack via <a href="https://docs.amd.com/">ROCm Information Portal</a>.</li>
<li>A ROCm enabled platform, more info <a href="https://rocm.github.io/install.html">here</a>.</li>
<li>Base software stack, which includes:</li>
<li>HIP - <ul>
<li>HIP and HCC libraries and header files.</li>
</ul>
</li>
<li>OpenCL - OpenCL libraries and header files.</li>
<li><a href="https://github.com/ROCmSoftwarePlatform/MIOpenGEMM">MIOpenGEMM</a> - enable various functionalities including transposed and dilated convolutions. </li>
<li>This is optional on the HIP backend, and required on the OpenCL backend.</li>
<li>Users can enable this library using the cmake configuration flag <code>-DMIOPEN_USE_MIOPENGEMM=On</code>, which is enabled by default when OpenCL backend is chosen.</li>
<li><a href="https://github.com/RadeonOpenCompute/rocm-cmake">ROCm cmake</a> - provide cmake modules for common build tasks needed for the ROCM software stack.</li>
<li><a href="http://half.sourceforge.net/">Half</a> - IEEE 754-based half-precision floating point library</li>
<li><a href="http://www.boost.org/">Boost</a> </li>
<li>MIOpen uses <code>boost-system</code> and <code>boost-filesystem</code> packages to enable persistent <a href="https://rocmsoftwareplatform.github.io/MIOpen/doc/html/cache.html">kernel cache</a></li>
<li>Version 1.79 is recommended, older version may need patches to work on newer systems, e.g. boost1{69,70,72} w/glibc-2.34</li>
<li><a href="https://sqlite.org/index.html">SQLite3</a> - reading and writing performance database</li>
<li>lbzip2 - multi-threaded compress or decompress utility</li>
<li><a href="https://github.com/ROCmSoftwarePlatform/MIOpenTensile">MIOpenTENSILE</a> - users can enable this library using the cmake configuration flag<code>-DMIOPEN_USE_MIOPENTENSILE=On</code>. (deprecated after ROCm 5.1.1)</li>
<li><a href="https://github.com/ROCmSoftwarePlatform/rocBLAS">rocBLAS</a> - AMD library for Basic Linear Algebra Subprograms (BLAS) on the ROCm platform.</li>
<li>Minimum version branch for pre-ROCm 3.5 <a href="https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/master-rocm-2.10">master-rocm-2.10</a></li>
<li>Minimum version branch for post-ROCm 3.5 <a href="https://github.com/ROCmSoftwarePlatform/rocBLAS/releases/tag/rocm-3.5.0">master-rocm-3.5</a></li>
<li><a href="https://github.com/ROCmSoftwarePlatform/llvm-project-mlir">MLIR</a> - (Multi-Level Intermediate Representation) with its MIOpen dialect to support and complement kernel development.</li>
<li><a href="https://github.com/ROCmSoftwarePlatform/composable_kernel">Composable Kernel</a> - C++ templated device library for GEMM-like and reduction-like operators.</li>
</ul>
<h2>Installing MIOpen with pre-built packages</h2>
<p>MIOpen can be installed on Ubuntu using <code>apt-get</code>.</p>
<p>For OpenCL backend: <code>apt-get install miopen-opencl</code></p>
<p>For HIP backend: <code>apt-get install miopen-hip</code></p>
<p>Currently both the backends cannot be installed on the same system simultaneously. If a different backend other than what currently exists on the system is desired, please uninstall the existing backend completely and then install the new backend.</p>
<h2>Installing MIOpen kernels package</h2>
<p>MIOpen provides an optional pre-compiled kernels package to reduce the startup latency. These precompiled kernels comprise a select set of popular input configurations and will expand in future release to contain additional coverage.</p>
<p>Note that all compiled kernels are locally cached in the folder <code>$HOME/.cache/miopen/</code>, so precompiled kernels reduce the startup latency only for the first execution of a neural network. Precompiled kernels do not reduce startup time on subsequent runs.</p>
<p>To install the kernels package for your GPU architecture, use the following command:</p>
<p><code>apt-get install miopenkernels-&lt;arch&gt;-&lt;num cu&gt;</code></p>
<p>Where <code>&lt;arch&gt;</code> is the GPU architecture ( for example, <code>gfx900</code>, <code>gfx906</code>, <code>gfx1030</code> ) and <code>&lt;num cu&gt;</code> is the number of CUs available in the GPU (for example 56 or 64 etc). </p>
<p>Not installing these packages would not impact the functioning of MIOpen, since MIOpen will compile these kernels on the target machine once the kernel is run. However, the compilation step may significantly increase the startup time for different operations.</p>
<p>The script <code>utils/install_precompiled_kernels.sh</code> provided as part of MIOpen automates the above process, it queries the user machine for the GPU architecture and then installs the appropriate package. It may be invoked as: </p>
<p><code>./utils/install_precompiled_kernels.sh</code></p>
<p>The above script depends on the <strong>rocminfo</strong> package to query the GPU architecture.</p>
<p>More info can be found <a href="https://github.com/ROCmSoftwarePlatform/MIOpen/blob/develop/doc/src/cache.md#installing-pre-compiled-kernels">here</a>.</p>
<h2>Installing the dependencies</h2>
<p>The dependencies can be installed with the <code>install_deps.cmake</code>, script: <code>cmake -P install_deps.cmake</code></p>
<p>This will install by default to <code>/usr/local</code> but it can be installed in another location with <code>--prefix</code> argument:
<code>cmake -P install_deps.cmake --prefix &lt;miopen-dependency-path&gt;</code>
An example cmake step can be:
<code>cmake -P install_deps.cmake --minimum --prefix /root/MIOpen/install_dir</code>
This prefix can used to specify the dependency path during the configuration phase using the <code>CMAKE_PREFIX_PATH</code>.</p>
<ul>
<li>
<p>MIOpen's HIP backend uses <a href="https://github.com/ROCmSoftwarePlatform/rocBLAS">rocBLAS</a> by default. Users can install rocBLAS minimum release by using <code>apt-get install rocblas</code>. To disable using rocBLAS set the configuration flag <code>-DMIOPEN_USE_ROCBLAS=Off</code>. rocBLAS is <em>not</em> available for the OpenCL backend.</p>
</li>
<li>
<p>MIOpen's OpenCL backend uses <a href="https://github.com/ROCmSoftwarePlatform/MIOpenGEMM">MIOpenGEMM</a> by default. Users can install MIOpenGEMM minimum release by using <code>apt-get install miopengemm</code>.</p>
</li>
</ul>
<h2>Building MIOpen from source</h2>
<h3>Configuring with cmake</h3>
<p>First create a build directory:</p>
<p><code>mkdir build; cd build;</code></p>
<p>Next configure cmake. The preferred backend for MIOpen can be set using the <code>-DMIOPEN_BACKEND</code> cmake variable. </p>
<h3>For the HIP backend (ROCm 3.5 and later), run:</h3>
<p>Set the C++ compiler to <code>clang++</code>.
<code>export CXX=&lt;location-of-clang++-compiler&gt;
cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="&lt;hip-installed-path&gt;;&lt;rocm-installed-path&gt;;&lt;miopen-dependency-path&gt;" ..</code></p>
<p>An example cmake step can be:
<code>export CXX=/opt/rocm/llvm/bin/clang++ &amp;&amp; \
cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;/root/MIOpen/install_dir" ..</code></p>
<p>Note: When specifying the path for the <code>CMAKE_PREFIX_PATH</code> variable, <strong>do not</strong> use the <code>~</code> shorthand for the user home directory.</p>
<h3>For OpenCL, run:</h3>
<p><code>cmake -DMIOPEN_BACKEND=OpenCL ..</code></p>
<p>The above assumes that OpenCL is installed in one of the standard locations. If not, then manually set these cmake variables: </p>
<p><code>cmake -DMIOPEN_BACKEND=OpenCL -DMIOPEN_HIP_COMPILER=&lt;hip-compiler-path&gt; -DOPENCL_LIBRARIES=&lt;opencl-library-path&gt; -DOPENCL_INCLUDE_DIRS=&lt;opencl-headers-path&gt; ..</code></p>
<p>And an example setting the dependency path for an envirnment in ROCm 3.5 and later:
<code>cmake -DMIOPEN_BACKEND=OpenCL -DMIOPEN_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/hip;/root/MIOpen/install_dir" ..</code></p>
<h3>Setting Up Locations</h3>
<p>By default the install location is set to '/opt/rocm', this can be set by using <code>CMAKE_INSTALL_PREFIX</code>:</p>
<p><code>cmake -DMIOPEN_BACKEND=OpenCL -DCMAKE_INSTALL_PREFIX=&lt;miopen-installed-path&gt; ..</code></p>
<h3>System Performance Database and User Database</h3>
<p>The default path to the System PerfDb is <code>miopen/share/miopen/db/</code> within install location. The default path to the User PerfDb is <code>~/.config/miopen/</code>. For development purposes, setting <code>BUILD_DEV</code> will change default path to both database files to the source directory:</p>
<p><code>cmake -DMIOPEN_BACKEND=OpenCL -DBUILD_DEV=On ..</code></p>
<p>Database paths can be explicitly customized by means of <code>MIOPEN_SYSTEM_DB_PATH</code> (System PerfDb) and <code>MIOPEN_USER_DB_PATH</code> (User PerfDb) cmake variables.</p>
<p>More information about the performance database can be found <a href="https://rocmsoftwareplatform.github.io/MIOpen/doc/html/perfdatabase.html">here</a>.</p>
<h3>Persistent Program Cache</h3>
<p>MIOpen by default caches the device programs in the location <code>~/.cache/miopen/</code>. In the cache directory there exists a directory for each version of MIOpen. Users can change the location of the cache directory during configuration using the flag <code>-DMIOPEN_CACHE_DIR=&lt;cache-directory-path&gt;</code>. </p>
<p>Users can also disable the cache during runtime using the environmental variable set as <code>MIOPEN_DISABLE_CACHE=1</code>. </p>
<h4>For MIOpen version 2.3 and earlier</h4>
<p>If the compiler changes, or the user modifies the kernels then the cache must be deleted for the MIOpen version in use; e.g., <code>rm -rf ~/.cache/miopen/&lt;miopen-version-number&gt;</code>. More information about the cache can be found <a href="https://rocmsoftwareplatform.github.io/MIOpen/doc/html/cache.html">here</a>.</p>
<h4>For MIOpen version 2.4 and later</h4>
<p>MIOpen's kernel cache directory is versioned so that users' cached kernels will not collide when upgrading from earlier version.</p>
<h3>Changing the cmake configuration</h3>
<p>The configuration can be changed after running cmake by using <code>ccmake</code>:</p>
<p><code>ccmake ..</code> <strong>OR</strong> <code>cmake-gui</code>: <code>cmake-gui ..</code></p>
<p>The <code>ccmake</code> program can be downloaded as the Linux package <code>cmake-curses-gui</code>, but is not available on windows.</p>
<h2>Building the library</h2>
<p>The library can be built, from the <code>build</code> directory using the 'Release' configuration:</p>
<p><code>cmake --build . --config Release</code> <strong>OR</strong> <code>make</code></p>
<p>And can be installed by using the 'install' target:</p>
<p><code>cmake --build . --config Release --target install</code> <strong>OR</strong> <code>make install</code></p>
<p>This will install the library to the <code>CMAKE_INSTALL_PREFIX</code> path that was set. </p>
<h2>Building the driver</h2>
<p>MIOpen provides an <a href="https://github.com/ROCmSoftwarePlatform/MIOpen/tree/master/driver">application-driver</a> which can be used to execute any one particular layer in isolation and measure performance and verification of the library. </p>
<p>The driver can be built using the <code>MIOpenDriver</code> target:</p>
<p><code>cmake --build . --config Release --target MIOpenDriver</code> <strong>OR</strong> <code>make MIOpenDriver</code></p>
<p>Documentation on how to run the driver is <a href="https://rocmsoftwareplatform.github.io/MIOpen/doc/html/driver.html">here</a>. </p>
<h2>Running the tests</h2>
<p>The tests can be run by using the 'check' target:</p>
<p><code>cmake --build . --config Release --target check</code> <strong>OR</strong> <code>make check</code></p>
<p>A single test can be built and ran, by doing:</p>
<p><code>cmake --build . --config Release --target test_tensor
./bin/test_tensor</code></p>
<h2>Formatting the code</h2>
<p>All the code is formatted using clang-format. To format a file, use:</p>
<p><code>clang-format-10 -style=file -i &lt;path-to-source-file&gt;</code></p>
<p>Also, githooks can be installed to format the code per-commit:</p>
<p><code>./.githooks/install</code></p>
<h2>Storing large file using Git LFS</h2>
<p>Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics with text pointers inside Git, while storing the file contents on a remote server. In MIOpen, we use git LFS to store the large files, such as the kernel database files (*.kdb) which are normally &gt; 0.5GB. Steps:</p>
<p>Git LFS can be installed and set up by:</p>
<p><code>sudo apt install git-lfs
git lfs install</code></p>
<p>In the Git repository that you want to use Git LFS, track the file type that you's like by (if the file type has been tracked, this step can be skipped):</p>
<p><code>git lfs track "*.file_type"
git add .gitattributes</code></p>
<p>Pull all or a single large file that you would like to update by:</p>
<p><code>git lfs pull --exclude=
or
git lfs pull --exclude= --include "filename"</code></p>
<p>Update the large files and push to the github by:</p>
<p><code>git add my_large_files
git commit -m "the message"
git push</code></p>
<h2>Installing the dependencies manually</h2>
<p>If Ubuntu v16 is used then the <code>Boost</code> packages can also be installed by:
<code>sudo apt-get install libboost-dev
sudo apt-get install libboost-system-dev
sudo apt-get install libboost-filesystem-dev</code></p>
<p><em>Note:</em> MIOpen by default will attempt to build with Boost statically linked libraries. If it is needed, the user can build with dynamically linked Boost libraries by using this flag during the configruation stage:
<code>-DBoost_USE_STATIC_LIBS=Off</code>
however, this is not recommended.</p>
<p>The <code>half</code> header needs to be installed from <a href="http://half.sourceforge.net/">here</a>. </p>
<h2>Using docker</h2>
<p>The easiest way is to use docker. You can build the top-level docker file:
<code>docker build -t miopen-image .</code></p>
<p>Then to enter the development environment use <code>docker run</code>, for example:
<code>docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device /dev/dri:/dev/dri:rw  --volume /dev/dri:/dev/dri:rw -v /var/lib/docker/:/var/lib/docker --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined miopen-image</code></p>
<p>Prebuilt docker images can be found on <a href="https://hub.docker.com/r/rocm/miopen/tags">ROCm's public docker hub here</a>.</p>
<h2>Citing MIOpen</h2>
<p>MIOpen's paper is freely available and can be accessed on arXiv:<br />
<a href="https://arxiv.org/abs/1910.00078">MIOpen: An Open Source Library For Deep Learning Primitives</a></p>
<h3>Citation BibTeX</h3>
<p><code>@misc{jeh2019miopen,
    title={MIOpen: An Open Source Library For Deep Learning Primitives},
    author={Jehandad Khan and Paul Fultz and Artem Tamazov and Daniel Lowell and Chao Liu and Michael Melesse and Murali Nandhimandalam and Kamil Nasyrov and Ilya Perminov and Tejash Shah and Vasilii Filippov and Jing Zhang and Jing Zhou and Bragadeesh Natarajan and Mayank Daga},
    year={2019},
    eprint={1910.00078},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}</code></p>
<h2>Porting from cuDNN to MIOpen</h2>
<p>The <a href="https://github.com/ROCmSoftwarePlatform/MIOpen/tree/develop/doc/src/MIOpen_Porting_Guide.md">porting
guide</a>
highlights the key differences between the current cuDNN and MIOpen APIs.</p>