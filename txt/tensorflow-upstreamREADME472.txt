<h1>TensorFlow Bazel Clang</h1>
<p>This is a specialized toolchain that uses an old Debian with a new Clang that
can cross compile to any x86_64 microarchitecture. It's intended to build Linux
binaries that only require the following ABIs:</p>
<ul>
<li>GLIBC_2.18</li>
<li>CXXABI_1.3.7 (GCC 4.8.3)</li>
<li>GCC_4.2.0</li>
</ul>
<p>Which are available on at least the following Linux platforms:</p>
<ul>
<li>Ubuntu 14+</li>
<li>CentOS 7+</li>
<li>Debian 8+</li>
<li>SuSE 13.2+</li>
<li>Mint 17.3+</li>
<li>Manjaro 0.8.11</li>
</ul>
<h1>System Install</h1>
<p>On Debian 8 (Jessie) Clang 6.0 can be installed as follows:</p>
<p><code>sh
cat &gt;&gt;/etc/apt/sources.list &lt;&lt;'EOF'
deb http://apt.llvm.org/jessie/ llvm-toolchain-jessie main
deb-src http://apt.llvm.org/jessie/ llvm-toolchain-jessie main
EOF
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
apt-key fingerprint |&amp; grep '6084 F3CF 814B 57C1 CF12  EFD5 15CF 4D18 AF4F 7421'
apt-get update
apt-get install clang lld</code></p>
<h1>Bazel Configuration</h1>
<p>This toolchain can compile TensorFlow in 2m30s on a 96-core Skylake GCE VM if
the following <code>.bazelrc</code> settings are added:</p>
<p>```
startup --host_jvm_args=-Xmx30G
startup --host_jvm_args=-Xms30G
startup --host_jvm_args=-XX:MaxNewSize=3g
startup --host_jvm_args=-XX:-UseAdaptiveSizePolicy
startup --host_jvm_args=-XX:+UseConcMarkSweepGC
startup --host_jvm_args=-XX:TargetSurvivorRatio=70
startup --host_jvm_args=-XX:SurvivorRatio=6
startup --host_jvm_args=-XX:+UseCMSInitiatingOccupancyOnly
startup --host_jvm_args=-XX:CMSFullGCsBeforeCompaction=1
startup --host_jvm_args=-XX:CMSInitiatingOccupancyFraction=75</p>
<p>build --jobs=100
build --local_resources=200000,100,100
build --crosstool_top=@local_config_clang6//clang6
build --noexperimental_check_output_files
build --nostamp
build --config=opt
build --noexperimental_check_output_files
build --copt=-march=native
build --host_copt=-march=native
```</p>
<h1>x86_64 Microarchitectures</h1>
<h2>Intel CPU Line</h2>
<ul>
<li>2003 P6 M SSE SSE2</li>
<li>2004 prescott SSE3 SSSE3 (-march=prescott)</li>
<li>2006 core X64 SSE4.1 (only on 45nm variety) (-march=core2)</li>
<li>2008 nehalem SSE4.2 VT-x VT-d (-march=nehalem)</li>
<li>2010 westmere CLMUL AES (-march=westmere)</li>
<li>2012 sandybridge AVX TXT (-march=sandybridge)</li>
<li>2012 ivybridge F16C MOVBE (-march=ivybridge)</li>
<li>2013 haswell AVX2 TSX BMI2 FMA (-march=haswell)</li>
<li>2014 broadwell RDSEED ADCX PREFETCHW (-march=broadwell - works on trusty
    gcc4.9)</li>
<li>2015 skylake SGX ADX MPX
    AVX-512<a href="-march=skylake / -march=skylake-avx512 - needs gcc7">xeon-only</a></li>
<li>2018 cannonlake AVX-512 SHA (-march=cannonlake - needs clang5)</li>
</ul>
<h2>Intel Low Power CPU Line</h2>
<ul>
<li>2013 silvermont SSE4.1 SSE4.2 VT-x (-march=silvermont)</li>
<li>2016 goldmont SHA (-march=goldmont - needs clang5)</li>
</ul>
<h2>AMD CPU Line</h2>
<ul>
<li>2003 k8 SSE SSE2 (-march=k8)</li>
<li>2005 k8 (Venus) SSE3 (-march=k8-sse3)</li>
<li>2008 barcelona SSE4a?! (-march=barcelona)</li>
<li>2011 bulldozer SSE4.1 SSE4.2 CLMUL AVX AES FMA4?! (-march=bdver1)</li>
<li>2011 piledriver FMA (-march=bdver2)</li>
<li>2015 excavator AVX2 BMI2 MOVBE (-march=bdver4)</li>
</ul>
<h2>Google Compute Engine Supported CPUs</h2>
<ul>
<li>2012 sandybridge 2.6gHz -march=sandybridge</li>
<li>2012 ivybridge 2.5gHz -march=ivybridge</li>
<li>2013 haswell 2.3gHz -march=haswell</li>
<li>2014 broadwell 2.2gHz -march=broadwell</li>
<li>2015 skylake 2.0gHz -march=skylake-avx512</li>
</ul>
<p>See: <a href="https://cloud.google.com/compute/docs/cpu-platforms">https://cloud.google.com/compute/docs/cpu-platforms</a></p>