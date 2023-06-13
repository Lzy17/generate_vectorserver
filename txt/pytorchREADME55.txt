<h1>Omniglot MAML examples</h1>
<p>In this directory we've provided some examples of training omniglot that reproduce the experiments from <a href="https://arxiv.org/abs/1703.03400">the original MAML paper</a>.</p>
<p>They can be run via <code>python {filename}</code>.</p>
<p><code>maml-omniglot-higher.py</code> uses the <a href="https://github.com/facebookresearch/higher">facebookresearch/higher</a> metalearning package and is the reference implementation. It runs all of its tasks sequentially.</p>
<p><code>maml-omniglot-transforms.py</code> uses functorch. It runs all of its tasks in parallel. In theory this should lead to some speedups, but we haven't finished implementing all the rules for vmap that would actually make training faster.</p>
<p><code>maml-omniglot-ptonly.py</code> is an implementation of <code>maml-omniglot-transforms.py</code> that runs all of its tasks sequentially (and also doesn't use the higher package).</p>