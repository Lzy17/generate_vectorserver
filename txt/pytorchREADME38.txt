<h1>Snapdragon NPE Support</h1>
<h2>Build</h2>
<p>Use the typical build_android script, but include a couple Cmake options to enable Snapdragon NPE:</p>
<pre><code>NPE_HEADERS=/path/to/snpe-1.2.2/include/
NPE_LOCATION=/path/to/snpe-1.2.2/lib/arm-android-gcc4.9/libSNPE.so
./scripts/build_android.sh -DUSE_SNPE=ON -DSNPE_LOCATION=$NPE_LOCATION -DSNPE_HEADERS=$NPE_HEADERS
</code></pre>
<p>this will enable the Snapdragon NPE Operator, which is a Caffe2 operator that can execute NPE <code>dlc</code> files.</p>
<h2>Usage</h2>
<p>Follow Qualcomm's instructions to convert a model into <code>dlc</code> format. You can then use the <code>dlc</code> as a Caffe2 operator in Python:</p>
<pre><code>with open('submodel.dlc', 'rb') as f:
    dlc = f.read()

op = core.CreateOperator('SNPE', ['data_in'], ['data_out'],
         arg=[
             utils.MakeArgument("model_buffer", dlc),
             utils.MakeArgument("input_name", "data") # Assuming DLC's first layer takes in 'data'
         ]
     )
</code></pre>
<p>and adding the operator to your model as you would normally.</p>
<h2>Debug</h2>
<p><code>libSNPE.so</code> is a shared library that is loaded at runtime.  You may need to specify the location of the library on your Android device when running standalone binaries.  The runtime assumes it will be able to <code>dlopen()</code> a file named <code>libSNPE.so</code> at the location specified by <code>gSNPELocation()</code>.  Either change this value at runtime or use an environment variable such as <code>LD_LIBRARY_PATH</code>.</p>
<p>You also need <code>libgnustl_shared.so</code> from Android NDK to be loaded in order to run standalone binaries. </p>