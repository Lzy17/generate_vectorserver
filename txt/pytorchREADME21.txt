<h1>clog: C-style (a-la printf) logging library</h1>
<p><a href="https://github.com/pytorch/cpuinfo/blob/master/deps/clog/LICENSE"><img alt="BSD (2 clause) License" src="https://img.shields.io/badge/License-BSD%202--Clause%20%22Simplified%22%20License-blue.svg" /></a></p>
<p>C-style library for logging errors, warnings, information notes, and debug information.</p>
<h2>Features</h2>
<ul>
<li>printf-style interface for formatting variadic parameters.</li>
<li>Separate functions for logging errors, warnings, information notes, and debug information.</li>
<li>Independent logging settings for different modules.</li>
<li>Logging to logcat on Android and stderr/stdout on other platforms.</li>
<li>Compatible with C99 and C++.</li>
<li>Covered with unit tests.</li>
</ul>
<h2>Example</h2>
<p>```c</p>
<h1>include <clog.h></h1>
<h1>ifndef MYMODULE_LOG_LEVEL</h1>
<pre><code>#define MYMODULE_LOG_LEVEL CLOG_DEBUG
</code></pre>
<h1>endif</h1>
<p>CLOG_DEFINE_LOG_DEBUG(mymodule_, "My Module", MYMODULE_LOG_LEVEL);
CLOG_DEFINE_LOG_INFO(mymodule_, "My Module", MYMODULE_LOG_LEVEL);
CLOG_DEFINE_LOG_WARNING(mymodule_, "My Module", MYMODULE_LOG_LEVEL);
CLOG_DEFINE_LOG_ERROR(mymodule_, "My Module", MYMODULE_LOG_LEVEL);</p>
<p>...</p>
<p>void some_function(...) {
    int status = ...
    if (status != 0) {
        mymodule_log_error(
            "something really bad happened: "
            "operation failed with status %d", status);
    }</p>
<pre><code>uint32_t expected_zero = ...
if (expected_zero != 0) {
    mymodule_log_warning(
        "something suspicious happened (var = %"PRIu32"), "
        "fall back to generic implementation", expected_zero);
}

void* usually_non_null = ...
if (usually_non_null == NULL) {
    mymodule_log_info(
        "something unusual, but common, happened: "
        "enabling work-around");
}

float a = ...
mymodule_log_debug("computed a = %.7f", a);
</code></pre>
<p>}
```</p>