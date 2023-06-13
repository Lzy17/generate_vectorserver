<p>This directory contains the useful tools.</p>
<h2>build_android.sh</h2>
<p>This script is to build PyTorch/Caffe2 library for Android. Take the following steps to start the build:</p>
<ul>
<li>set ANDROID_NDK to the location of ndk</li>
</ul>
<p><code>bash
export ANDROID_NDK=YOUR_NDK_PATH</code></p>
<ul>
<li>run build_android.sh
```bash</li>
</ul>
<h1>in your PyTorch root directory</h1>
<p>bash scripts/build_android.sh
```
If succeeded, the libraries and headers would be generated to build_android/install directory. You can then copy these files from build_android/install to your Android project for further usage.</p>
<p>You can also override the cmake flags via command line, e.g., following command will also compile the executable binary files:
<code>bash
bash scripts/build_android.sh -DBUILD_BINARY=ON</code></p>
<h2>build_ios.sh</h2>
<p>This script is to build PyTorch/Caffe2 library for iOS, and can only be performed on macOS. Take the following steps to start the build:</p>
<ul>
<li>Install Xcode from App Store, and configure "Command Line Tools" properly on Xcode.</li>
<li>Install the dependencies:</li>
</ul>
<p><code>bash
brew install cmake automake libtool</code></p>
<ul>
<li>run build_ios.sh
```bash</li>
</ul>
<h1>in your PyTorch root directory</h1>
<p>bash scripts/build_ios.sh
```
If succeeded, the libraries and headers would be generated to build_ios/install directory. You can then copy these files  to your Xcode project for further usage.</p>