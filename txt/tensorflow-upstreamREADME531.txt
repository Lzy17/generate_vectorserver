<h1>GPU delegate compatibility database</h1>
<p>This package provides data and code for deciding if the GPU delegate is
supported on a specific Android device.</p>
<h2>Customizing the database</h2>
<ul>
<li>Convert from checked-in flatbuffer to json by running <code>flatc -t --raw-binary
    --strict-json database.fbs -- gpu_compatibility.bin</code></li>
<li>Edit the json</li>
<li>Convert from json to flatbuffer <code>flatc -b database.fbs --
    gpu_compatibility.json</code></li>
<li>Rebuild ../../../java:tensorflow-lite-gpu</li>
</ul>