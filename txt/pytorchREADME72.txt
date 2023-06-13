<h2>FX Pass Infrastructure</h2>
<p>This folder contains the pass infrastructure and passes for transforming fx.Graph.</p>
<h2>Code Structure</h2>
<ul>
<li><a href="infra">infra</a> - Common infrastructure, such as PassManager, PassBase<ul>
<li><a href="infra/partitioner.py">partitioner.py</a> - backend agnostic FX graph partitioner</li>
</ul>
</li>
<li><a href="utils">utils</a> - Utility classes and functions<ul>
<li><a href="utils/common.py">common.py</a> - common utility functions</li>
<li><a href="utils/fuser_utils.py">fuser_utis.py</a> - utility functions for fusing list of nodes into a single node</li>
</ul>
</li>
<li><a href="dialect">dialect</a> - dialect specific passes<ul>
<li><a href="dialect/common">common</a> - common passes that can be shared by all dialects<ul>
<li><a href="dialect/common/cse_pass.py">cse_pass.py</a> - a CSE pass</li>
</ul>
</li>
<li><a href="dialect/aten">aten</a> - aten dialect specific passes</li>
<li><a href="dialect/prims">prims</a> - prim dialect specific passes</li>
</ul>
</li>
<li><a href="backends">backends</a> - Backend specific passes<ul>
<li><a href="backends/nvfuser">nvfuser</a> - passes for nvfuser<ul>
<li><a href="backends/nvfuser/operator_support.py">operator_support.py</a> - nvFuser supported ops</li>
</ul>
</li>
</ul>
</li>
<li><a href="conversion">conversion</a> - Conversion passes between dialects</li>
</ul>