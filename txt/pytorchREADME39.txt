<h1>libopencl-stub</h1>
<p>A stub opencl library that dynamically dlopen/dlsyms opencl implementations at runtime based on environment variables. Will be useful when opencl implementations are installed in non-standard paths (say pocl on android)</p>
<p>LIBOPENCL_SO_PATH      -- Path to opencl so that will be searched first</p>
<p>LIBOPENCL_SO_PATH_2    -- Searched second</p>
<p>LIBOPENCL_SO_PATH_3    -- Searched third</p>
<p>LIBOPENCL_SO_PATH_4    -- Searched fourth</p>
<p>Default paths will be searched otherwise</p>