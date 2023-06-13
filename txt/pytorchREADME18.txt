<h1>QNNPACK</h1>
<p>QNNPACK (Quantized Neural Networks PACKage) is a mobile-optimized library for low-precision high-performance neural network inference. QNNPACK provides implementation of common neural network operators on quantized 8-bit tensors.</p>
<p>QNNPACK is not intended to be directly used by machine learning researchers; instead it provides low-level performance primitives for high-level deep learning frameworks. As of today, QNNPACK is integrated in <a href="https://github.com/pytorch/pytorch">PyTorch 1.0</a> with Caffe2 graph representation.</p>
<h2>Operator Coverage</h2>
<p>Currently implemented and planned for implementation operators are below:</p>
<ul>
<li>[x] 2D Convolution</li>
<li>[x] 2D Deconvolution</li>
<li>[x] Channel Shuffle</li>
<li>[x] Fully Connected</li>
<li>[ ] Locally Connected</li>
<li>[x] 2D Max Pooling</li>
<li>[x] 2D Average Pooling</li>
<li>[x] Global Average Pooling</li>
<li>[x] Sigmoid</li>
<li>[x] TanH</li>
<li>[x] Leaky ReLU</li>
<li>[x] Hardsigmoid</li>
<li>[x] Hardswish</li>
<li>[x] Clamp (can be used for ReLU, ReLU6 if it is not fused in another operator)</li>
<li>[x] SoftArgMax (aka SoftMax)</li>
<li>[ ] Group Normalization</li>
</ul>
<h2>Building</h2>
<p>QNNPACK provides standard CMake-based build scripts.</p>
<h3>Native compilation</h3>
<p>Users are recommended to use <code>scripts/build-local.sh</code> script to build QNNPACK for the host machine.</p>
<h3>Cross-compilation for Android</h3>
<p>To cross-compile for Android, set <code>$ANDROID_NDK</code> environment variable (where <code>$ANDROID_NDK</code> is the path to Android NDK directory, e.g. <code>/opt/android-ndk-r15c</code>) and use one of the scripts from the table below:</p>
<p>| ABI         | Build script                     | Restrictions               |
| ----------- | ---------------------------------| -------------------------- |
| armeabi-v7a | <code>scripts/build-android-armv7.sh</code> | Requires CPU with ARM NEON |
| arm64-v8a   | <code>scripts/build-android-arm64.sh</code> |                            |
| x86         | <code>scripts/build-android-x86.sh</code>   |                            |</p>
<p>Notes:
- On <strong>armeabi-v7a</strong> <code>pytorch_qnnp_initialize</code> will fail with <code>pytorch_qnnp_status_unsupported_hardware</code> if the mobile CPU does not support ARM NEON. Don't set <code>-DANDROID_ARM_NEON=1</code> for QNNPACK compilation as it can make <code>pytorch_qnnp_initialize</code> crash on CPUs without ARM NEON.</p>
<h3>Cross-compilation for iOS</h3>
<p>To cross-compile for iOS, clone <a href="https://github.com/leetal/ios-cmake">ios-cmake</a>, and set <code>$IOS_CMAKE_TOOLCHAIN_FILE</code> environment variable (where <code>$IOS_CMAKE_TOOLCHAIN_FILE</code> is the path to <code>ios.toolchain.cmake</code> file in <a href="https://github.com/leetal/ios-cmake">ios-cmake</a>), and use one of the scripts from the table below:</p>
<p>| Architecture | Build script                  | Notes                     |
| ------------ | ----------------------------- | ------------------------- |
| armv7        | <code>scripts/build-ios-armv7.sh</code>  | iPhone 3GS/4/4S           |
| armv7        | <code>scripts/build-ios-armv7s.sh</code> | iPhone 5 and newer        |
| arm64        | <code>scripts/build-ios-arm64.sh</code>  | iPhone 5S and newer       |
| arm64e       | <code>scripts/build-ios-arm64e.sh</code> | iPhone XS/XR              |
| i386         | <code>scripts/build-ios-i386.sh</code>   | iPhone Simulator (32-bit) |
| x86_64       | <code>scripts/build-ios-x86_64.sh</code> | iPhone Simulator (64-bit) |</p>
<h2>End-to-End Benchmarking</h2>
<p>Caffe2 backend of PyTorch 1.0 natively integrates QNNPACK, and provides a <a href="https://github.com/caffe2/models/tree/master/mobilenet_v2_quantized">pre-trained quantized MobileNet v2 model</a>. Below are instructions for benchmarking this model end-to-end with QNNPACK.</p>
<h3>Raspberry Pi 2 or 3</h3>
<p>```bash</p>
<h1>Clone PyTorch 1.0 repo</h1>
<p>git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch</p>
<h1>Optional: update QNNPACK submodule to latest revision</h1>
<p>git submodule update --remote third_party/QNNPACK</p>
<h1>Build Caffe2 (including binaries) for the host system</h1>
<h1>Use only 1 thread for build to avoid out-of-memory failures</h1>
<p>MAX_JOBS=1 scripts/build_local.sh -DBUILD_BINARY=ON -DBUILD_PYTHON=OFF \
    -DUSE_OBSERVERS=OFF -DUSE_DISTRIBUTED=OFF</p>
<h1>Download model weights</h1>
<p>wget https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2_1.0_224_quant/init_net.pb</p>
<h1>Download model graph</h1>
<p>wget https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2_1.0_224_quant/predict_net.pb</p>
<h1>Run speed benchmark with 50 warm-up iterations and 10 measurement iterations</h1>
<p>build/bin/speed_benchmark --net predict_net.pb --init_net init_net.pb \
    --input data --input_dims 1,3,224,224 --input_type float \
    --warmup 50 --iter 10
```</p>
<h3>ARMv7 (32-bit) Android</h3>
<p>```bash</p>
<h1>Clone PyTorch 1.0 repo</h1>
<p>git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch</p>
<h1>Optional: update QNNPACK submodule to latest revision</h1>
<p>git submodule update --remote third_party/QNNPACK</p>
<h1>Build Caffe2 (including binaries) for Android, and push to device</h1>
<p>scripts/build_android.sh -DANDROID_TOOLCHAIN=clang -DBUILD_BINARY=ON
adb push build_android/bin/speed_benchmark /data/local/tmp/speed_benchmark</p>
<h1>Download model weights and copy them to Android device</h1>
<p>wget https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2_1.0_224_quant/init_net.pb
adb push init_net.pb /data/local/tmp/init_net.pb</p>
<h1>Download model graph and copy it to Android device</h1>
<p>wget https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2_1.0_224_quant/predict_net.pb
adb push predict_net.pb /data/local/tmp/predict_net.pb</p>
<h1>Run speed benchmark with 50 warm-up iterations and 10 measurement iterations</h1>
<p>adb shell /data/local/tmp/speed_benchmark \
    --net /data/local/tmp/predict_net.pb \
    --init_net /data/local/tmp/init_net.pb \
    --input data --input_dims 1,3,224,224 --input_type float \
    --warmup 50 --iter 10
```</p>
<h3>ARM64 (64-bit) Android</h3>
<p>```bash</p>
<h1>Clone PyTorch 1.0 repo</h1>
<p>git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch</p>
<h1>Optional: update QNNPACK submodule to latest revision</h1>
<p>git submodule update --remote third_party/QNNPACK</p>
<h1>Build Caffe2 (including binaries) for Android, and push to device</h1>
<p>scripts/build_android.sh -DANDROID_ABI=arm64-v8a -DANDROID_TOOLCHAIN=clang -DBUILD_BINARY=ON
adb push build_android/bin/speed_benchmark /data/local/tmp/speed_benchmark</p>
<h1>Download model weights and copy them to Android device</h1>
<p>wget https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2_1.0_224_quant/init_net.pb
adb push init_net.pb /data/local/tmp/init_net.pb</p>
<h1>Download model graph and copy it to Android device</h1>
<p>wget https://s3.amazonaws.com/download.caffe2.ai/models/mobilenet_v2_1.0_224_quant/predict_net.pb
adb push predict_net.pb /data/local/tmp/predict_net.pb</p>
<h1>Run speed benchmark with 50 warm-up iterations and 10 measurement iterations</h1>
<p>adb shell /data/local/tmp/speed_benchmark \
    --net /data/local/tmp/predict_net.pb \
    --init_net /data/local/tmp/init_net.pb \
    --input data --input_dims 1,3,224,224 --input_type float \
    --warmup 50 --iter 10
```</p>
<h3>PEP (Performance Evaluation Platform) Method</h3>
<p><a href="https://github.com/facebook/FAI-PEP">Facebook AI Performance Evaluation Platform</a> is a framework and backend agnostic benchmarking platform to compare machine learning inferencing runtime metrics on a set of models and a variety of backends.</p>
<p>We use PEP to produce the results we have in our <a href="https://code.fb.com/ml-applications/qnnpack/">blog</a></p>
<p>With an ARMv7 device connected:</p>
<p>```bash</p>
<h1>Clone PyTorch 1.0 repo</h1>
<p>mkdir ~/Code &amp;&amp; cd ~/Code
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch</p>
<h1>Optional: update QNNPACK submodule to latest revision</h1>
<p>git submodule update --remote third_party/QNNPACK</p>
<h1>Clone PEP repo</h1>
<p>cd ~/Code
git clone --recursive https://github.com/facebook/FAI-PEP.git aibench
cd aibench</p>
<h1>Run PEP benchmark with cool specifications. Try changing that cmd with more specifications!</h1>
<h1>First time compile could take 20+ minutes</h1>
<p>./benchmarking/run_bench.py \
  --platform android \
  -b ~/Code/aibench/specifications/models/caffe2/mobilenet_v2/mobilenet_v2_quant.json \
  --platform android --repo_dir ~/Code/pytorch \
  --frameworks_dir ~/Code/aibench/specifications/frameworks --framework caffe2
```</p>
<h2>Acknowledgements</h2>
<p>QNNPACK is developed by Marat Dukhan, Yiming Wu, Hao Lu, and Bert Maher. We thank Andrew Tulloch and Yangqing Jia for advice during the development of QNNPACK.</p>
<h2>License</h2>
<p>QNNPACK is BSD licensed, as found in the <a href="LICENSE"><code>LICENSE</code></a> file.</p>