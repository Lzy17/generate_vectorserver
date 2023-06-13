<h2>Miniz</h2>
<p>Miniz is a lossless, high performance data compression library in a single source file that implements the zlib (RFC 1950) and Deflate (RFC 1951) compressed data format specification standards. It supports the most commonly used functions exported by the zlib library, but is a completely independent implementation so zlib's licensing requirements do not apply. Miniz also contains simple to use functions for writing .PNG format image files and reading/writing/appending .ZIP format archives. Miniz's compression speed has been tuned to be comparable to zlib's, and it also has a specialized real-time compressor function designed to compare well against fastlz/minilzo.</p>
<h2>Usage</h2>
<p>Please use the files from the <a href="https://github.com/richgel999/miniz/releases">releases page</a> in your projects. Do not use the git checkout directly! The different source and header files are <a href="https://www.sqlite.org/amalgamation.html">amalgamated</a> into one <code>miniz.c</code>/<code>miniz.h</code> pair in a build step (<code>amalgamate.sh</code>). Include <code>miniz.c</code> and <code>miniz.h</code> in your project to use Miniz.</p>
<h2>Features</h2>
<ul>
<li>MIT licensed</li>
<li>A portable, single source and header file library written in plain C. Tested with GCC, clang and Visual Studio.</li>
<li>Easily tuned and trimmed down by defines</li>
<li>A drop-in replacement for zlib's most used API's (tested in several open source projects that use zlib, such as libpng and libzip).</li>
<li>Fills a single threaded performance vs. compression ratio gap between several popular real-time compressors and zlib. For example, at level 1, miniz.c compresses around 5-9% better than minilzo, but is approx. 35% slower. At levels 2-9, miniz.c is designed to compare favorably against zlib's ratio and speed. See the miniz performance comparison page for example timings.</li>
<li>Not a block based compressor: miniz.c fully supports stream based processing using a coroutine-style implementation. The zlib-style API functions can be called a single byte at a time if that's all you've got.</li>
<li>Easy to use. The low-level compressor (tdefl) and decompressor (tinfl) have simple state structs which can be saved/restored as needed with simple memcpy's. The low-level codec API's don't use the heap in any way.</li>
<li>Entire inflater (including optional zlib header parsing and Adler-32 checking) is implemented in a single function as a coroutine, which is separately available in a small (~550 line) source file: miniz_tinfl.c</li>
<li>A fairly complete (but totally optional) set of .ZIP archive manipulation and extraction API's. The archive functionality is intended to solve common problems encountered in embedded, mobile, or game development situations. (The archive API's are purposely just powerful enough to write an entire archiver given a bit of additional higher-level logic.)</li>
</ul>
<h2>Known Problems</h2>
<ul>
<li>No support for encrypted archives. Not sure how useful this stuff is in practice.</li>
<li>Minimal documentation. The assumption is that the user is already familiar with the basic zlib API. I need to write an API wiki - for now I've tried to place key comments before each enum/API, and I've included 6 examples that demonstrate how to use the module's major features.</li>
</ul>
<h2>Special Thanks</h2>
<p>Thanks to Alex Evans for the PNG writer function. Also, thanks to Paul Holden and Thorsten Scheuermann for feedback and testing, Matt Pritchard for all his encouragement, and Sean Barrett's various public domain libraries for inspiration (and encouraging me to write miniz.c in C, which was much more enjoyable and less painful than I thought it would be considering I've been programming in C++ for so long).</p>
<p>Thanks to Bruce Dawson for reporting a problem with the level_and_flags archive API parameter (which is fixed in v1.12) and general feedback, and Janez Zemva for indirectly encouraging me into writing more examples.</p>
<h2>Patents</h2>
<p>I was recently asked if miniz avoids patent issues. miniz purposely uses the same core algorithms as the ones used by zlib. The compressor uses vanilla hash chaining as described <a href="http://www.gzip.org/zlib/rfc-deflate.html#algorithm">here</a>. Also see the <a href="http://www.gzip.org/#faq11">gzip FAQ</a>. In my opinion, if miniz falls prey to a patent attack then zlib/gzip are likely to be at serious risk too.</p>
<p><a href="https://travis-ci.org/uroni/miniz"><img alt="Build Status" src="https://travis-ci.org/uroni/miniz.svg?branch=master" /></a></p>