<h1>Accelerator allowlisting</h1>
<p>Experimental library and tools for determining whether an accelerator engine
works well on a given device, and for a given model.</p>
<h2>Platform-agnostic, Android-first</h2>
<p>Android-focused, since the much smaller set of configurations on iOS means there
is much less need for allowlisting on iOS.</p>
<h2>Not just for TfLite</h2>
<p>This code lives in the TfLite codebase, since TfLite is the first open-source
customer. It is however meant to support other users (direct use of NNAPI,
mediapipe).</p>