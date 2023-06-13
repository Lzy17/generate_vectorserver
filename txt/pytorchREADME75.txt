<h1>Data Sparsifier Benchmarking using the DLRM Model</h1>
<h2>Introduction</h2>
<p>The objective of this exercise is to use the data sparsifier to prune the embedding bags of the <a href="https://github.com/facebookresearch/dlrm">DLRM Model</a> and observe the following -</p>
<ol>
<li><strong>Disk usage savings</strong>: Savings in model size after pruning.</li>
<li><strong>Model Quality</strong>: How and by how much does performance deteriorate after pruning the embedding bags?</li>
<li><strong>Model forward time</strong>: Can we speed up the model forward time by utilizing the sparsity? Specifically, can we introduce torch.sparse interim to reduce number of computations.</li>
</ol>
<h2>Scope</h2>
<p>The <a href="https://github.com/pytorch/pytorch/blob/master/torch/ao/sparsity/_experimental/data_sparsifier/data_norm_sparsifier.py">DataNormSparsifier</a> is used to sparsify the embeddings of the DLRM model. The model is sparsified for all the combinations of -
1. Sparsity Levels: [0.0, 0.1, 0.2, ... 0.9, 0.91, 0.92, ... 0.99, 1.0]
2. Sparse Block shapes: (1,1) and (1,4)
3. Norm: L1 and L2</p>
<h2>Dataset</h2>
<p>The benchmarks are created for the dlrm model on the Kaggle CriteoDataset which can be downloaded from <a href="https://ailab.criteo.com/ressources/">here</a> or <a href="https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310/1">here</a>.</p>
<h2>Results</h2>
<ol>
<li><strong>Disk Usage</strong>: Introducing sparsity in the embeddings reduces file size after compression. The compressed model size goes down from 1.9 GB to 150 MB after 100% sparsity.</li>
</ol>
<p><img src="./images/disk_savings.png" align="center" height="250" width="400" ><img src="./images/accuracy.png" align="right" height="250" width="400" ></p>
<ol>
<li>
<p><strong>Model Quality</strong>: The model accuracy decreases slowly with sparsity levels. Even at 90% sparsity levels, the model accuracy decreases only by 2%.</p>
</li>
<li>
<p><strong>Model forward time</strong>: Sparse coo tensors are introduced on the features before feeding into the top layer of the dlrm model. Post that, we perform a sparse <code>torch.mm</code> with the first linear weight of the top layer.
The takeaway is that the dlrm model with sparse coo tensor is slower (roughly 2x). This is because even though the sparsity levels are high in the embedding weights, the interaction step between the dense and sparse features increases the sparsity levels. Hence, creating sparse coo tensor on this not so sparse features actually slows down the model.</p>
</li>
</ol>
<p><img src="./images/forward_time.png" height="250" width="400" ></p>
<h2>Setup</h2>
<p>The benchmark codes depend on the <a href="https://github.com/facebookresearch/dlrm">DLRM codebase</a>.
1. Clone the dlrm git repository
2. Download the dataset from <a href="https://ailab.criteo.com/ressources/">here</a> or <a href="https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310/1">here</a>
3. The DLRM model can be trained using the following script
```</p>
<h1>Make sure you go into the file and make sure that the path to dataset is correct.</h1>
<p>./bench/dlrm_s_criteo_kaggle.sh --save-model=./models/criteo_model.ckpt [--use-gpu]</p>
<h1>This should also dump kaggleAdDisplayChallenge_processed.npz in the path where data is present</h1>
<p>```</p>
<ol>
<li>Copy the scripts data sparsifier benchmark scripts into to the dlrm directory.</li>
</ol>
<h2>Scripts to run each experiment.</h2>
<h3><strong>Disk savings</strong></h3>
<p><code>python evaluate_disk_savings.py --model-path=&lt;path_to_model_checkpoint&gt; --sparsified-model-dump-path=&lt;path_to_dump_sparsified_models&gt;</code></p>
<p>Running this script should dump
* sparsified model checkpoints: model is sparsified for all the
    combinations of sparsity levels, block shapes and norms and dumped.</p>
<ul>
<li><code>sparse_model_metadata.csv</code>: This contains the compressed file size and path info for all the sparsified models. This file will be used for other experiments</li>
</ul>
<h3><strong>Model Quality</strong></h3>
<p><code>python evaluate_model_metrics.py --raw-data-file=&lt;path_to_raw_data_txt_file&gt; --processed-data-file=&lt;path_to_kaggleAdDisplayChallenge_processed.npz&gt; --sparse-model-metadata=&lt;path_to_sparse_model_metadata_csv&gt;</code>
Running this script should dump <code>sparse_model_metrics.csv</code> that contains evaluation metrics for all sparsified models.</p>
<h3><strong>Model forward time</strong>:</h3>
<p><code>python evaluate_forward_time.py --raw-data-file=&lt;path_to_raw_data_txt_file&gt; --processed-data-file=&lt;path_to_kaggleAdDisplayChallenge_processed.npz&gt; --sparse-model-metadata=&lt;path_to_sparse_model_metadata_csv&gt;</code>
Running this script should dump <code>dlrm_forward_time_info.csv</code> that contains forward time for all sparsified models with and without torch.sparse in the forward pass.</p>
<h2>Requirements</h2>
<p>pytorch (latest)</p>
<p>scikit-learn</p>
<p>numpy</p>
<p>pandas</p>
<h2>Machine specs to create benchmark</h2>
<p>AI AWS was used to run everything i.e. training the dlrm model and running data sparsifier benchmarks.</p>
<p>Machine: AI AWS</p>
<p>Instance Type: p4d.24xlarge</p>
<p>GPU: A100</p>
<h2>Future work</h2>
<ol>
<li>
<p><strong>Evaluate memory savings</strong>: The idea is to use torch.sparse tensors to store weights of the embedding bags so that the model memory consumption improves. This will be possible once the embedding bags starts supporting torch.sparse backend.</p>
</li>
<li>
<p><strong>Sparsifying activations</strong>: Use activation sparsifier to sparsify the activations of the dlrm model. The idea is to sparsify the features before feeding to the top dense layer (sparsify <code>z</code> <a href="https://github.com/facebookresearch/dlrm/blob/11afc52120c5baaf0bfe418c610bc5cccb9c5777/dlrm_s_pytorch.py#L595">here</a>).</p>
</li>
</ol>