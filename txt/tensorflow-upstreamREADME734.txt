<h1>Public API of mlir_replay</h1>
<p>This contains protocol buffers and utilities that can be reused for other
debugging tools:</p>
<ol>
<li><strong>The compiler trace proto</strong>: A record of the state of the IR after each
    compilation pass</li>
<li>A compiler instrumentation to create the above proto.</li>
<li><strong>The execution trace proto</strong>: A record of SSA values as the IR is executed</li>
<li>Utilities for working with the above protos.</li>
</ol>