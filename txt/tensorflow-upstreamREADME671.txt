<h1>TensorFlow Lite C++ minimal example</h1>
<p>This example shows how you can build a simple TensorFlow Lite application.</p>
<h4>Step 1. Install CMake tool</h4>
<p>It requires CMake 3.16 or higher. On Ubuntu, you can simply run the following
command.</p>
<p><code>sh
sudo apt-get install cmake</code></p>
<p>Or you can follow
<a href="https://cmake.org/install/">the official cmake installation guide</a></p>
<h4>Step 2. Clone TensorFlow repository</h4>
<p><code>sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src</code></p>
<h4>Step 3. Create CMake build directory and run CMake tool</h4>
<p><code>sh
mkdir minimal_build
cd minimal_build
cmake ../tensorflow_src/tensorflow/lite/examples/minimal</code></p>
<h4>Step 4. Build TensorFlow Lite</h4>
<p>In the minimal_build directory,</p>
<p><code>sh
cmake --build . -j</code></p>