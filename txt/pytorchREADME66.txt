<p>c10/core/impl provides headers for functionality that is only needed in very
<em>specific</em> use-cases (e.g., you are defining a new device type), which are
generally only needed by C10 or PyTorch code.  If you are an ordinary end-user,
you <strong>should not</strong> use headers in this folder.  We permanently give NO
backwards-compatibility guarantees for implementations in this folder.</p>
<p>Compare with <a href="../../util">c10/util</a>, which provides functionality that is not
directly related to being a deep learning library (e.g., C++20 polyfills), but
may still be generally useful and visible to users.</p>
<p>(We don't call this c10/detail, because the detail namespace convention is for
<em>header private</em> details.  However, c10::impl may be utilized from external
headers; it simply indicates that the functionality is not for end users.)</p>