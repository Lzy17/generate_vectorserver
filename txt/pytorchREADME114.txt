<h1>Build</h1>
<p><code>mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make -j</code></p>
<h1>Test</h1>
<p><code>./sinh_example</code></p>