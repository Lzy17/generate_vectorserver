<h1>Android</h1>
<h2>Demo applications and tutorials</h2>
<p>Demo applications with code walk-through can be find in <a href="https://github.com/pytorch/android-demo-app">this github repo</a>.</p>
<h2>Publishing</h2>
<h5>Release</h5>
<p>Release artifacts are published to jcenter:</p>
<p>```
repositories {
    jcenter()
}</p>
<h1>lite interpreter build</h1>
<p>dependencies {
    implementation 'org.pytorch:pytorch_android_lite:1.10.0'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:1.10.0'
}</p>
<h1>full jit build</h1>
<p>dependencies {
    implementation 'org.pytorch:pytorch_android:1.10.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.10.0'
}
```</p>
<h5>Nightly</h5>
<p>Nightly(snapshots) builds are published every night from <code>master</code> branch to <a href="https://oss.sonatype.org/#nexus-search;quick~pytorch_android">nexus sonatype snapshots repository</a></p>
<p>To use them repository must be specified explicitly:
```
repositories {
    maven {
        url "https://oss.sonatype.org/content/repositories/snapshots"
    }
}</p>
<h1>lite interpreter build</h1>
<p>dependencies {
    ...
    implementation 'org.pytorch:pytorch_android_lite:1.12.0-SNAPSHOT'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:1.12.0-SNAPSHOT'
    ...
}</p>
<h1>full jit build</h1>
<p>dependencies {
    ...
    implementation 'org.pytorch:pytorch_android:1.12.0-SNAPSHOT'
    implementation 'org.pytorch:pytorch_android_torchvision:1.12.0-SNAPSHOT'
    ...
}
<code>``
The current nightly(snapshots) version is the value of</code>VERSION_NAME<code>in</code>gradle.properties<code>in current folder, at this moment it is</code>1.8.0-SNAPSHOT`.</p>
<h2>Building PyTorch Android from Source</h2>
<p>In some cases you might want to use a local build of pytorch android, for example you may build custom libtorch binary with another set of operators or to make local changes.</p>
<p>For this you can use <code>./scripts/build_pytorch_android.sh</code> script.
<code>git clone https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
bash ./scripts/build_pytorch_android.sh</code></p>
<p>The workflow contains several steps:</p>
<p>1. Build libtorch for android for all 4 android abis (armeabi-v7a, arm64-v8a, x86, x86_64)</p>
<p>2. Create symbolic links to the results of those builds:
<code>android/pytorch_android/src/main/jniLibs/${abi}</code> to the directory with output libraries
<code>android/pytorch_android/src/main/cpp/libtorch_include/${abi}</code> to the directory with headers. These directories are used to build <code>libpytorch.so</code> library that will be loaded on android device.</p>
<p>3. And finally run <code>gradle</code> in <code>android/pytorch_android</code> directory with task <code>assembleRelease</code></p>
<p>Script requires that Android SDK, Android NDK and gradle are installed.
They are specified as environment variables:</p>
<p><code>ANDROID_HOME</code> - path to <a href="https://developer.android.com/studio/command-line/sdkmanager.html">Android SDK</a></p>
<p><code>ANDROID_NDK</code> - path to <a href="https://developer.android.com/studio/projects/install-ndk">Android NDK</a>. It's recommended to use NDK 21.x.</p>
<p><code>GRADLE_HOME</code> - path to <a href="https://gradle.org/releases/">gradle</a></p>
<p>After successful build you should see the result as aar file:</p>
<p><code>$ find pytorch_android/build/ -type f -name *aar
pytorch_android/build/outputs/aar/pytorch_android.aar
pytorch_android_torchvision/build/outputs/aar/pytorch_android.aar</code></p>
<p>It can be used directly in android projects, as a gradle dependency:
```
allprojects {
    repositories {
        flatDir {
            dirs 'libs'
        }
    }
}</p>
<p>dependencies {
    implementation(name:'pytorch_android', ext:'aar')
    implementation(name:'pytorch_android_torchvision', ext:'aar')
    ...
    implementation 'com.facebook.soloader:nativeloader:0.10.5'
    implementation 'com.facebook.fbjni:fbjni-java-only:0.2.2'
}
<code>``
We also have to add all transitive dependencies of our aars.
As</code>pytorch_android<code>[depends](https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/build.gradle#L76-L77) on</code>'com.facebook.soloader:nativeloader:0.10.5'<code>and</code>'com.facebook.fbjni:fbjni-java-only:0.2.2'<code>, we need to add them.
(In case of using maven dependencies they are added automatically from</code>pom.xml`).</p>
<p>You can check out <a href="https://github.com/pytorch/pytorch/blob/master/android/test_app/app/build.gradle">test app example</a> that uses aars directly.</p>
<h2>Linking to prebuilt libtorch library from gradle dependency</h2>
<p>In some cases, you may want to use libtorch from your android native build.
You can do it without building libtorch android, using native libraries from PyTorch android gradle dependency.
For that, you will need to add the next lines to your gradle build.
```
android {
...
    configurations {
       extractForNativeBuild
    }
...
    compileOptions {
        externalNativeBuild {
            cmake {
                arguments "-DANDROID_STL=c++_shared"
            }
        }
    }
...
    externalNativeBuild {
        cmake {
            path "CMakeLists.txt"
        }
    }
}</p>
<p>dependencies {
    extractForNativeBuild('org.pytorch:pytorch_android:1.10.0')
}</p>
<p>task extractAARForNativeBuild {
    doLast {
        configurations.extractForNativeBuild.files.each {
            def file = it.absoluteFile
            copy {
                from zipTree(file)
                into "$buildDir/$file.name"
                include "headers/<strong>"
                include "jni/</strong>"
            }
        }
    }
}</p>
<p>tasks.whenTaskAdded { task -&gt;
  if (task.name.contains('externalNativeBuild')) {
    task.dependsOn(extractAARForNativeBuild)
  }
}
```</p>
<p>pytorch_android aar contains headers to link in <code>headers</code> folder and native libraries in <code>jni/$ANDROID_ABI/</code>.
As PyTorch native libraries use <code>ANDROID_STL</code> - we should use <code>ANDROID_STL=c++_shared</code> to have only one loaded binary of STL.</p>
<p>The added task will unpack them to gradle build directory.</p>
<p>In your native build you can link to them adding these lines to your CMakeLists.txt:</p>
<p>```</p>
<h1>Relative path of gradle build directory to CMakeLists.txt</h1>
<p>set(build_DIR ${CMAKE_SOURCE_DIR}/build)</p>
<p>file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android<em>.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android</em>.aar/jni/${ANDROID_ABI}")</p>
<p>set(BUILD_SUBDIR ${ANDROID_ABI})
target_include_directories(${PROJECT_NAME} PRIVATE
  ${PYTORCH_INCLUDE_DIRS}
)</p>
<p>find_library(PYTORCH_LIBRARY pytorch_jni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)</p>
<p>find_library(FBJNI_LIBRARY fbjni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)</p>
<p>target_link_libraries(${PROJECT_NAME}
  ${PYTORCH_LIBRARY})
  ${FBJNI_LIBRARY})</p>
<p><code>``
If your CMakeLists.txt file is located in the same directory as your build.gradle,</code>set(build_DIR ${CMAKE_SOURCE_DIR}/build)` should work for you. But if you have another location of it, you may need to change it.</p>
<p>After that, you can use libtorch C++ API from your native code.
```</p>
<h1>include <string></h1>
<h1>include <ATen/NativeFunctions.h></h1>
<h1>include <torch/script.h></h1>
<p>namespace pytorch_testapp_jni {
namespace {
    struct JITCallGuard {
      c10::InferenceMode guard;
      torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
    };
}</p>
<p>void loadAndForwardModel(const std::string&amp; modelPath) {
  JITCallGuard guard;
  torch::jit::Module module = torch::jit::load(modelPath);
  module.eval();
  torch::Tensor t = torch::randn({1, 3, 224, 224});
  c10::IValue t_out = module.forward({t});
}
}
```</p>
<p>To load torchscript model for mobile we need some special setup which is placed in <code>struct JITCallGuard</code> in this example. It may change in future, you can track the latest changes keeping an eye in our <a href="[https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/cpp/pytorch_jni_jit.cpp#L28">pytorch android jni code</a></p>
<p><a href="https://github.com/pytorch/pytorch/tree/master/android/test_app">Example of linking to libtorch from aar</a></p>
<h2>PyTorch Android API Javadoc</h2>
<p>You can find more details about the PyTorch Android API in the <a href="https://pytorch.org/javadoc/">Javadoc</a>.</p>