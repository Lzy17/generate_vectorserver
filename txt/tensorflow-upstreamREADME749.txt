<h1>TensorFlow Op CodeGen Machinery (Experimental)</h1>
<h2>Usage</h2>
<p><code>usage: generate_cpp  [flags]  OpName1 [OpName2 ...]
Flags:
    --help=false                        bool    Print this help message.
    --category=""                       string  Category for generated ops (e.g. 'math', 'array').
    --namespace=""                      string  Compact C++ namespace, default is 'tensorflow::ops'.
    --output_dir=""                     string  Directory into which output files will be generated.
    --source_dir=""                     string  The tensorflow root directory, e.g. 'tensorflow/' for in-source include paths. Any path underneath the tensorflow root is also accepted.
    --api_dirs=""                       string  Comma-separated list of directories containing API definitions.</code></p>
<h2>Design</h2>
<h3>Generator Framework</h3>
<p>The generator framework is a loose Model/View/Controller arrangement:</p>
<p>The <em>Model</em> classes live in the <strong><em>model/</em></strong> directory. They are representations
of the <code>OpDef</code> and <code>ApiDef</code> protos, normalized and resolved.</p>
<blockquote>
<p><em>For example, an <code>OpDef</code> proto's <code>ArgDef</code> members contain a type string, which
must be dereferenced to an <code>AttrDef</code> by name to determine its type. This
<code>AttrDef</code> proto message in turn contains a type string which may need to be
parsed as "list(type)". Other <code>AttrDef</code> messages are not types, but instead
argument-like modifiers. In contrast, the generator model <code>ArgSpec</code> contains a
resolved <code>ArgType</code> which provides a boolean <code>is_list()</code> method directly, and
the model <code>OpSpec</code> provides a list of only the argument-like attributes. In
addition to convenience, this should aid consistency between generated code in
each target language.</em></p>
</blockquote>
<p>The <em>Controller</em> is in the <strong><em>common/</em></strong> directory. It is the workhorse used by
the language generators; it digests the Op registry and API definitions to build
the model and provides utilities for the language generators.</p>
<p>The <em>View</em> and rendering classes map the language-independent Model classes
(<code>OpSpec</code>, <code>ArgSpec</code>, <code>AttrSpec</code>, etc.) to language-specific <code>SourceCode</code>. The
framework does not impose any design on the language-specific generators, but
provides some utilities, and the C++ generator is a complete example.</p>
<h3>C++ Generator</h3>
<p>The <code>CppGenerator</code> class is the interface to the <code>cpp/</code> language directory.
Given a config, it can generate source code for a .cc or .h file as a string or
write it to a target file.</p>
<p>The <code>CppFileRenderer</code> is the main renderer used by the generator; it renders an
entire file. The <code>CppConfig</code> defines if it is operating in header or source
mode.</p>
<p>"Views" are stateless and intended to be low-level building blocks: a direct
language-specific representation of the model classes. For example, an <code>ArgView</code>
is initialized from an <code>ArgSpec</code> (which was created initially from an <code>ArgDef</code>
proto message). Where they may have some similar methods between the model and
view, the view methods are language-specific.</p>
<p>For instance, the C++ generator's <code>ArgView::VariableName()</code> method is an
language-formatted name usable as a variable representing the model <code>ArgSpec</code>
object. In contrast, the <code>ArgSpec::name()</code> method in the model refers to the
canonical name of the object in the proto.</p>
<p>Where views are a representation of the <em>input</em> model, in the C++ generator,
"renderers" then use these views to build the <em>output</em> <code>SourceCode</code>; Renderers
understand the language at the statement/directive level and target a functional
section of the output, such as a block comment or an entire method or file.</p>
<p>Other differences between views and renderers:</p>
<ul>
<li>Renderers are stateful, modifying a referenced SourceCode. Views are
    stateless and their public methods are all const, returning strings.</li>
<li>Renderers are context-dependent, e.g. a method signature will include
    default values when in "declaration" mode but not "definition" mode. A view
    of some argument object simply knows its default value and does not care the
    context.</li>
<li>In terms of dependencies, <code>Renderers</code> use <code>Views</code> and other <code>Renderers</code>.
    However, <code>Renderers</code> do <strong>not</strong> reference the model directly (e.g.
    <code>OpSpec</code>). This is because if a renderer needs to reference part of the
    model, it should get a language specific representation.</li>
</ul>
<h3>Extending to Additional Languages</h3>
<p>The design for the C++ generator should apply to other languages, and the
underlying generator framework (the model and controller) try to be agnostic. In
fact, some elements of the C++ design could be formalized (such as the
rendering/view framework) or re-used (e.g. <code>cpp:Renderer</code> could likely be shared
with C and Java as a common C-style language renderer base class).</p>
<p>Abstracted and condensed from the C++ generator, the overall control flow could
be described as follows:</p>
<p>From main() in <em>generate_lang_main.cc</em>:</p>
<ul>
<li>Call <code>tensorflow::port::InitMain</code> and parse any flags</li>
<li>Initialize config objects (e.g. <code>PathConfig</code>, <code>LangConfig</code> from flags)</li>
<li>Initialize a new <code>LangGenerator</code> from these config objects</li>
<li>Call this generator to create/write <code>SourceCode</code> to a file</li>
</ul>
<p>In class <code>LangGenerator</code> in <em>lang_generator.cc</em>:</p>
<ul>
<li>Initialize a new <code>Controller</code> from the config objects</li>
<li>Call this controller to build the Op models (<code>OpSpec</code>)</li>
<li>Initialize a new language-specific <code>View</code> for each model object</li>
<li>Create a blank <code>SourceCode</code> rendering target (for each output file)</li>
<li>Initialize a new <code>LangFileRenderer</code> from this target source code, the model
    <code>View</code> objects, and config objects</li>
<li>Call this renderer to generate the target <code>SourceCode</code></li>
</ul>
<p>The dependencies are as follows:</p>
<ul>
<li><code>lang::Generator</code> depends on <code>Controller</code>, <code>Model</code>, <code>lang::Renderers</code>,
    <code>lang::Views</code></li>
<li><code>lang::Renderer</code> depends on <code>lang::View</code> (and <code>lang::Renderer</code> peers)</li>
<li><code>lang::View</code> depends on the model (e.g. <code>OpSpec</code>) (and <code>lang::View</code> peers)</li>
</ul>