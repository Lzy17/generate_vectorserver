<h1>NNAPI Support Library</h1>
<p>Files in this directory are a copy of NNAPI Support Library
<a href="https://cs.android.com/android/platform/superproject/+/master:packages/modules/NeuralNetworks/shim_and_sl/;drc=629cea610b447266b1e6b01e4cb6a952dcb56e7e">files</a>
in AOSP.</p>
<p>The files had to be modified to make them work in TF Lite context. Here is the
list of differences from the AOSP version:</p>
<ul>
<li><code>#include</code> directives use fully-qualified paths.</li>
<li><code>#pragma once</code> directives are changed to header guards. Android paths in
    header guards are changed to TF Lite paths.</li>
<li><code>tensorflow/lite/nnapi/NeuralNetworksTypes.h</code> is used for definitions of
    NNAPI types instead of
    <a href="https://cs.android.com/android/_/android/platform/packages/modules/NeuralNetworks/+/6f0a05b9abdfe0d17afe0269c5329340809175b5:runtime/include/NeuralNetworksTypes.h;drc=a62e56b26b7382a62c5aa0e5964266eba55853d8"><code>NeuralNetworksTypes.h</code> from AOSP</a>.</li>
<li><code>loadNnApiSupportLibrary(...)</code> is using <code>tensorflow/lite/minimal_logging.h</code>
    for logging on errors.</li>
<li><code>SupportLibrary.h</code> declarations are wrapped into <code>tflite::nnapi</code> namespace.</li>
<li><code>__BEGIN_DECLS</code> and <code>__END_DECLS</code> are changed to explicit <code>extern "C"</code>
    blocks.</li>
<li>Copyright notice is changed to the one used in Tensorflow project.</li>
</ul>