<h1>Tensorflow C SavedModel API</h1>
<h2>Overview</h2>
<p>These are the new experimental C SavedModel APIs for loading and running
SavedModels in a TF2-idiomatic fashion. See
<a href="https://github.com/tensorflow/community/pull/207">RFC 207</a> for additional
context.</p>
<p>The directory structure is as follows:</p>
<p>```none
saved_model/</p>
<p>public/</p>
<p>internal/</p>
<p>core/</p>
<p>```</p>
<h2>saved_model/public</h2>
<p><code>saved_model/public</code> is intended to house <em>only the public headers</em> of the
SavedModel C API.</p>
<p>These headers:</p>
<ol>
<li>
<p>declare opaque C types (like <code>TF_SavedModel</code>),</p>
</li>
<li>
<p>declare the functions that operate on these types (like <code>TF_LoadSavedModel</code>).</p>
</li>
</ol>
<p>Once they leave experimental, these APIs should be considered stable for use
by external clients.</p>
<p>These headers are in a separate directory to make it obvious to clients which
headers they should depend on, and which headers are implementation details.
Separating these public headers by directory also allow future programmatic
checks to ensure that TF public headers only <code>#include</code> other public TF headers.</p>
<h2>saved_model/internal</h2>
<p><code>saved_model/internal</code> is the "glue" between the C API and the internal C++
implementation.</p>
<p>Its role is to:</p>
<ol>
<li>
<p>implement the C API functions declared in <code>saved_model/public</code></p>
</li>
<li>
<p>define the C API types declared in <code>saved_model/public</code></p>
</li>
</ol>
<p>The files fulfilling 1. are named <code>*.cc</code> (eg: <code>concrete_function.cc</code>), while
the files fulfilling 2. are <code>*type.h</code> (eg: <code>concrete_function_type.h</code>).</p>
<p>The headers exposing the internal implementation of the opaque C types are only
visible to other implementors of the C API. This is similar to how other
TF C API implementations use <code>tf_status_internal.h</code> (to extract the underlying
<code>tensorflow::Status</code>). All other targets in this directory are private.</p>
<h2>saved_model/core</h2>
<p><code>saved_model/core</code> contains pure C++ "Classes" underlying the C API types
in <code>saved_model/public/</code>. These are implementation
details subject to change, and have limited visibility to implementors only.
This is the bottom-most layer of the <code>C++ -&gt; C -&gt; C++</code> sandwich.</p>