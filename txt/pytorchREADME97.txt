<h2>Autograd</h2>
<p>Autograd is a hotspot for PyTorch performance, so most of the heavy lifting is
implemented in C++. This implies that we have to do some shuffling between
Python and C++; and in general, we want data to be in a form that is convenient
to manipulate from C++.</p>
<p>Our general model is that for any key data type that autograd manipulates,
there are two implementations: a C++ type and a Python object type.  For
example, consider variables in autograd: we have both <code>Variable</code> in <code>variable.h</code>
(the C++ type) and <code>THPVariable</code> in <code>python_variable.h</code> (the Python type.)
(By the way, THP stands for TorcH Python, not to be confused with THPP, TorcH
C++).  <code>Variable</code> contains the payload of a variable, while <code>THPVariable</code> just
contains a <code>shared_ptr</code> reference to <code>Variable</code>, as well as references to other
Python objects which the Python runtime needs to know about.  A lot of
data accessor implementations in <code>python_variable.cpp</code> simply reach through
to the underlying <code>Variable</code> and return the appropriate value.</p>
<p>The most complicated application of this principle is Function, which also
supports users implementing custom behavior in Python.  We have the following
classes:</p>
<ul>
<li><code>Node</code> in <code>function.h</code>, the C++ type.</li>
<li><code>THPFunction</code> in <code>python_function.h</code>, the Python object type.  In
  <code>python_function.cpp</code>, you can see the boilerplate that tells the Python
  interpreter about this object.</li>
<li><code>PyNode</code> in <code>python_function.h</code>, a subclass of <code>Node</code> which forwards
  <code>apply</code> to a Python <code>THPFunction</code>. (NOT a Python object, despite its name!)</li>
</ul>
<p>Outside of <code>PyNode</code>, the C++ objects largely avoid referencing Python
objects (there are a few exceptions, like <code>pyobj</code> in <code>Variable</code>, and
<code>PyNode</code>, whose whole point is to let C++ call into Python). And <code>pyobj</code>
in <code>Node</code> to ensure uniqueness of the associated python wrapper (if it exists).</p>