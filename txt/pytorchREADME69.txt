<h2>PyTorch for iOS</h2>
<h3>Cocoapods Developers</h3>
<p>PyTorch is now available via Cocoapods, to integrate it to your project, simply add the following line to your <code>Podfile</code> and run <code>pod install</code></p>
<p><code>ruby
pod 'LibTorch-Lite'</code></p>
<h3>Import the library</h3>
<p>For Objective-C developers, simply import the umbrella header</p>
<p>```</p>
<h1>import <LibTorch-Lite.h></h1>
<p>```</p>
<p>For Swift developers, you need to create an Objective-C class as a bridge to call the C++ APIs. We highly recommend you to follow the <a href="https://github.com/pytorch/ios-demo-app/tree/master/PyTorchDemo">Image Classification</a> demo where you can find out how C++, Objective-C and Swift work together.</p>
<h3>Disable Bitcode</h3>
<p>Since PyTorch is not yet built with bitcode support, you need to disable bitcode for your target by selecting the <strong>Build Settings</strong>, searching for <strong>Enable Bitcode</strong> and set the value to <strong>No</strong>.</p>
<h2>LICENSE</h2>
<p>PyTorch is BSD-style licensed, as found in the LICENSE file.</p>