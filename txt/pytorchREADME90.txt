<p>The <a href="https://github.com/pytorch/pytorch/tree/master/torch/utils/data/datapipes"><code>datapipes</code></a> folder holds the implementation of the <code>IterDataPipe</code> and <code>MapDataPipe</code>.</p>
<p>This document serves as an entry point for DataPipe implementation.</p>
<h2>Implementing DataPipe</h2>
<p>For the sake of an example, let us implement an <code>IterDataPipe</code> to apply a callable over data under <a href="https://github.com/pytorch/pytorch/tree/master/torch/utils/data/datapipes/iter"><code>iter</code></a>.
For <code>MapDataPipe</code>, please take reference from files in <a href="https://github.com/pytorch/pytorch/tree/master/torch/utils/data/datapipes/map">map</a> folder and implement the corresponding <code>__getitem__</code> method.</p>
<h3>Naming</h3>
<p>The naming convention for DataPipe is Operation-er and with suffix of <code>IterDataPipe</code> because each DataPipe behaves like a container to apply the operation to data yielded from the source DataPipe.
And, when importing the DataPipe into <code>iter</code> module under <code>datapipes</code>, each DataPipe will be aliased as Op-er without the suffix of <code>IterDataPipe</code>.
Please check <a href="https://github.com/pytorch/pytorch/blob/master/torch/utils/data/datapipes/iter/__init__.py"><code>__init__.py</code></a> in <code>iter</code> module for how we aliasing each DataPipe class.
Like the example of <code>IterDataPipe</code> to map a function, we are going to name it as <code>MapperIterDataPipe</code> and alias it as <code>iter.Mapper</code> under <code>datapipes</code>.</p>
<h3>Constructor</h3>
<p>As DataSet now constructed by a stack of DataPipe-s, each DataPipe normally takes a source DataPipe as the first argument.
<code>py
class MapperIterDataPipe(IterDataPipe):
    def __init__(self, dp, fn):
        super().__init__()
        self.dp = dp
        self.fn = fn</code>
Note:
- Avoid loading data from the source DataPipe in <code>__init__</code> function, in order to support lazy data loading and save memory.
- If <code>IterDataPipe</code> instance holds data in memory, please be ware of the in-place modification of data. When second iterator is created from the instance, the data may have already changed. Please take <a href="https://github.com/pytorch/pytorch/blob/master/torch/utils/data/datapipes/iter/utils.py"><code>IterableWrapper</code></a> class as reference to <code>deepcopy</code> data for each iterator.</p>
<h3>Iterator</h3>
<p>For <code>IterDataPipe</code>, an <code>__iter__</code> function is needed to consume data from the source <code>IterDataPipe</code> then apply operation over the data before yield.
```py
class MapperIterDataPipe(IterDataPipe):
    ...</p>
<pre><code>def __iter__(self):
    for d in self.dp:
        yield self.fn(d)
</code></pre>
<p>```</p>
<h3>Length</h3>
<p>In the most common cases, as the example of <code>MapperIterDataPipe</code> above, the <code>__len__</code> method of DataPipe should return the length of source DataPipe.
Take care that <code>__len__</code> must be computed dynamically, because the length of source data-pipes might change after initialization (for example if sharding is applied).</p>
<p>```py
class MapperIterDataPipe(IterDataPipe):
    ...</p>
<pre><code>def __len__(self):
    return len(self.dp)
</code></pre>
<p><code>``
Note that</code><strong>len</strong><code>method is optional for</code>IterDataPipe<code>.
Like</code>CSVParserIterDataPipe<code>in the [Using DataPipe sector](#using-datapipe),</code><strong>len</strong>` is not implemented because the size of each file streams is unknown for us before loading it.</p>
<p>Besides, in some special cases, <code>__len__</code> method can be provided, but it would either return an integer length or raise Error depending on the arguments of DataPipe.
And, the Error is required to be <code>TypeError</code> to support Python's build-in functions like <code>list(dp)</code>.
Please check NOTE [ Lack of Default <code>__len__</code> in Python Abstract Base Classes ] for detailed reason in PyTorch.</p>
<h3>Registering DataPipe with functional API</h3>
<p>Each DataPipe can be registered to support functional API using the decorator <code>functional_datapipe</code>.
<code>py
@functional_datapipe("map")
class MapperIterDataPipe(IterDataPipe):
    ...</code>
Then, the stack of DataPipe can be constructed in functional-programming manner.
```py</p>
<blockquote>
<blockquote>
<blockquote>
<p>import torch.utils.data.datapipes as dp
datapipes1 = dp.iter.FileOpener(['a.file', 'b.file']).map(fn=decoder).shuffle().batch(2)</p>
<p>datapipes2 = dp.iter.FileOpener(['a.file', 'b.file'])
datapipes2 = dp.iter.Mapper(datapipes2)
datapipes2 = dp.iter.Shuffler(datapipes2)
datapipes2 = dp.iter.Batcher(datapipes2, 2)
<code>``
In the above example,</code>datapipes1<code>and</code>datapipes2<code>represent the exact same stack of</code>IterDataPipe`-s.</p>
</blockquote>
</blockquote>
</blockquote>
<h2>Using DataPipe</h2>
<p>For example, we want to load data from CSV files with the following data pipeline:
- List all csv files
- Load csv files
- Parse csv file and yield rows</p>
<p>To support the above pipeline, <code>CSVParser</code> is registered as <code>parse_csv_files</code> to consume file streams and expand them as rows.
```py
@functional_datapipe("parse_csv_files")
class CSVParserIterDataPipe(IterDataPipe):
    def <strong>init</strong>(self, dp, **fmtparams):
        self.dp = dp
        self.fmtparams = fmtparams</p>
<pre><code>def __iter__(self):
    for filename, stream in self.dp:
        reader = csv.reader(stream, **self.fmtparams)
        for row in reader:
            yield filename, row
</code></pre>
<p><code>Then, the pipeline can be assembled as following:</code>py</p>
<blockquote>
<blockquote>
<blockquote>
<p>import torch.utils.data.datapipes as dp</p>
<p>FOLDER = 'path/2/csv/folder'
datapipe = dp.iter.FileLister([FOLDER]).filter(fn=lambda filename: filename.endswith('.csv'))
datapipe = dp.iter.FileOpener(datapipe, mode='rt')
datapipe = datapipe.parse_csv_files(delimiter=' ')</p>
<p>for d in datapipe: # Start loading data
...     pass
```</p>
</blockquote>
</blockquote>
</blockquote>