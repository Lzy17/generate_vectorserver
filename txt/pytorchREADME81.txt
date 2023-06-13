<h1>ModelReport</h1>
<h2>Model Report Class in Fx Workflow</h2>
<blockquote>
<p>⚠️ <em>While the example below uses the Fx Workflow, the use of the ModelReport class </em><em>does not depend</em><em> on the Fx Workflow to work</em>.
 The requirements are detector dependent.
 Most detectors require a <strong>traceable GraphModule</strong>, but some (ex. <code>PerChannelDetector</code>) require just an <code>nn.Module</code>.</p>
</blockquote>
<h4>Typical Fx Workflow</h4>
<ul>
<li>Initialize model &rarr; Prepare model &rarr; Callibrate model &rarr; Convert model &rarr; ...</li>
</ul>
<h4>Fx Workflow with ModelReport</h4>
<ul>
<li>Initialize model &rarr; Prepare model &rarr; <strong>Add detector observers</strong> &rarr; Callibrate model &rarr; <strong>Generate report</strong> &rarr; <strong>Remove detector observers</strong> &rarr; Convert model &rarr; ...</li>
</ul>
<blockquote>
<p>⚠️ <strong>You can only prepare and remove observers once with a given ModelReport Instance</strong>: Be very careful here!</p>
</blockquote>
<h2>Usage</h2>
<p>This snippet should be ready to copy, paste, and use with the exception of a few small parts denoted in <code>#TODO</code> comments</p>
<p>```python</p>
<h1>prep model</h1>
<p>qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()
model = Model() # TODO define model
example_input = torch.randn((*args)) # TODO get example data for callibration
prepared_model = quantize_fx.prepare_fx(model, qconfig_mapping, example_input)</p>
<h1>create ModelReport instance and insert observers</h1>
<p>detector_set = set([DynamicStaticDetector()]) # TODO add all desired detectors
model_report = ModelReport(model, detector_set)
ready_for_callibrate = model_report.prepare_detailed_callibration()</p>
<h1>callibrate model and generate report</h1>
<p>ready_for_callibrate(example_input) # TODO run callibration of model with relevant data
reports = model_report.generate_model_report(remove_inserted_observers=True)
for report_name in report.keys():
    text_report, report_dict = reports[report_name]
    print(text_report, report_dict)</p>
<h1>Optional: we get a ModelReportVisualizer instance to do any visualizations desired</h1>
<p>mod_rep_visualizer = tracer_reporter.generate_visualizer()
mod_rep_visualizer.generate_table_visualization() # shows collected data as a table</p>
<h1>TODO updated qconfig based on suggestions</h1>
<p>```</p>
<p>There is a tutorial in the works that will walk through a full usage of the ModelReport API.
This tutorial will show the ModelReport API being used on toy model in both an Fx Graph Mode workflow and an alterative workflow with just a traceable model.
This README will be updated with a link to the tutorial upon completion of the tutorial.</p>
<h1>Key Modules Overview</h1>
<h2>ModelReport Overview</h2>
<p>The <code>ModelReport</code> class is the primary class the user will be interacting with in the ModelReport workflow.
There are three primary methods to be familiar with when using the ModelReport class:</p>
<ul>
<li><code>__init__(self, model: GraphModule, desired_report_detectors: Set[DetectorBase])</code> constructor that takes in instances of the model we wish to generate report for (must be traceable GraphModule) and desired detectors and stores them.
This is so that we can keep track of where we want to insert observers on a detector by detector basis and also keep track of which detectors to generate reports for.</li>
<li><code>prepare_detailed_calibration(self)</code> &rarr; <code>GraphModule</code> inserts observers into the locations specified by each detector in the model.
It then returns the GraphModule with the detectors inserted into both the regular module structure as well as the node structure.</li>
<li><code>generate_model_report(self, remove_inserted_observers: bool)</code> &rarr; <code>Dict[str, Tuple[str, Dict]]</code> uses callibrated GraphModule to optionally removes inserted observers, and generate, for each detector the ModelReport instance was initialized with:</li>
<li>A string-based report that is easily digestable and actionable explaining the data collected by relevant observers for that detector</li>
<li>A dictionary containing statistics collected by the relevant observers and values calculated by the detector for further analysis or plotting</li>
</ul>
<h2>ModelReportVisualizer Overview</h2>
<p>After you have generated reports using the <code>ModelReport</code> instance,
you can visualize some of the collected statistics using the <code>ModelReportVisualizer</code>.
To get a <code>ModelReportVisualizer</code> instance from the <code>ModelReport</code> instance,
call <code>model_report.generate_visualizer()</code>.</p>
<p>When you first create the <code>ModelReportVisualizer</code> instance,
it reorganizes the reports so instead of being in a:</p>
<p><code>report_name
|
-- module_fqn
   |
   -- feature_name
      |
      -- feature value</code></p>
<p>format, it will instead be in a:
<code>-- module_fqn [ordered]
   |
   -- feature_name
      |
      -- feature value</code></p>
<p>Essentially, all the informations for each of the modules are consolidated across the different reports.
Moreover, the modules are kept in the same chronological order they would appear in the model's <code>forward()</code> method.</p>
<p>Then, when it comes to the visualizer, there are two main things you can do:
1. Call <code>mod_rep_visualizer.generate_filtered_tables()</code> to get a table of values you can manipulate
2. Call one of the generate visualization methods, which don't return anything but generate an output
  - <code>mod_rep_visualizer.generate_table_visualization()</code> prints out a neatly formatted table
  - <code>mod_rep_visualizer.generate_plot_visualization()</code> and <code>mod_rep_visualizer.generate_histogram_visualization()</code>
  output plots.</p>
<p>For both of the two things listed above, you can filter the data by either <code>module_fqn</code> or by <code>feature_name</code>.
To get a list of all the modules or features, you can call <code>mod_rep_visualizer.get_all_unique_module_fqns()</code>
and <code>mod_rep_visualizer.get_all_unique_feature_names()</code> respectively.
For the features, because some features are not plottable, you can set the flag to only get plottable features
in the aformentioned <code>get_all_unique_feature_names</code> method.</p>
<h2>Detector Overview</h2>
<p>The main way to add functionality to the ModelReport API is to add more Detectors.
Detectors each have a specific focus in terms of the type of information they collect.
For example, the <code>DynamicStaticDetector</code> figures out whether Dynamic or Static Quantization is appropriate for different layers.
Meanwhile, the <code>InputWeightEqualizationDetector</code> determines whether Input-Weight Equalization should be applied for each layer.</p>
<h3>Requirements to Implement A Detector</h3>
<p>All Detectors inherit from the <code>DetectorBase</code> class, and all of them (including any custom detectors you create) will need to implement 3 methods:
- <code>determine_observer_insert_points(self, model)</code> -&gt; <code>Dict</code>: determines which observers you want to insert into a model to gather statistics and where in the model.
All of them return a dictionary mapping unique observer fully qualified names (fqns), which is where we want to insert them, to a dictionary of location and argument information in the format:</p>
<p><code>python
return_dict = {
    "[unique_observer_fqn_of_insert_location]" :
    {
        "target_node" -&gt; the node we are trying to observe with this observer (torch.fx.node.Node),
        "insert_observer" -&gt; the initialized observer we wish to insert (ObserverBase),
        "insert_post" -&gt; True if this is meant to be a post-observer for target_node, False if pre-observer,
        "observer_args" -&gt; The arguments that are meant to be passed into the observer,
    }
}</code>
- <code>get_detector_name(self)</code> -&gt; <code>str</code>: returns the name of the detector.
You should give your detector a unique name different from existing detectors.
- <code>generate_detector_report(self, model)</code> -&gt; <code>Tuple[str, Dict[str, Any]]</code>: generates a report based on the information the detector is trying to collect.
This report consists of both a text-based report as well as a dictionary of collected and calculated statistics.
This report is returned to the <code>ModelReport</code> instance, which will then compile all the reports of all the Detectors requested by the user.</p>
<h2>ModelReportObserver Overview</h2>
<p>As seen in the <a href="#requirements-to-implement-a-detector">requirements to implement a detector section</a>, one of the key parts of implementing a detector is to specify what <code>Observer</code> we are trying to insert.
All the detectors in the ModelReport API use the <a href="https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/_model_report/model_report_observer.py"><code>ModelReportObserver</code></a>.
While the core purpose of many observers in PyTorch's Quantization API is to collect min / max information to help determine quantization parameters, the <code>ModelReportObserver</code> collects additional statistics.</p>
<p>The statistics collected by the <code>ModelReportObserver</code> include:
- Average batch activation range
- Epoch level activation range
- Per-channel min / max values
- Ratio of 100th percentile to some <em>n</em>th percentile
- Number of constant value batches to pass through each channel</p>
<p>After the <code>ModelReportObserver</code> collects the statistics above during the callibration process, the detectors then extract the information they need to generate their reports from the relevant observers.</p>
<h3>Using Your Own Observer</h3>
<p>If you wish to implement your own custom Observer to use with the ModelReport API for your own custom detector, there are a few things to keep in mind.
- Make sure your detector inherits from <a href="https://www.internalfb.com/code/fbsource/[20eb160510847bd24bf21a5b95092c160642155f]/fbcode/caffe2/torch/ao/quantization/observer.py?lines=122"><code>torch.ao.quantization.observer.ObserverBase</code></a>
- In the custom detector class, come up with a descriptive and unique <code>PRE_OBSERVER_NAME</code> (and/or <code>POST_OBSERVER_NAME</code>) so that you can generate a fully qualified name (fqn) for each observer that acts a key in the returned dictionary described <a href="#requirements-to-implement-a-detector">here</a>
  - <a href="https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/_model_report/detector.py#L958">Code Example</a>
- In the <code>determine_observer_insert_points()</code> method in your detector, initialize your custom Observer and add it to the returned dictionary described <a href="#requirements-to-implement-a-detector">here</a>
  - <a href="https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/_model_report/detector.py#L1047">Code Example</a></p>
<p>Since you are also implementing your own detector in this case, it is up to you to determine where your observers should be placed in the model, and what type of information you wish to extract from them to generate your report.</p>
<h1>Folder Structure</h1>
<p>./: the main folder all the model report code is under
- <code>__init__.py</code>: File to mark ModelReport as package directory
- <code>detector.py</code>: File containing Detector classes
  - Contains <code>DetectorBase</code> class which all detectors inherit from
  - Contains several implemented detectors including:
    - <code>PerChannelDetector</code>
    - <code>DynamicStaticDetector</code>
    - <code>InputWeightEqualizationDetector</code>
    - <code>OutlierDetector</code>
- <code>model_report_observer.py</code>: File containing the <code>ModelReportObserver</code> class
  - Primary observer inserted by Detectors to collect necessary information to generate reports
- <code>model_report_visualizer.py</code>: File containing the <code>ModelReportVisualizer</code> class
  - Reorganizes reports generated by the <code>ModelReport</code> class to be:
    1. Ordered by module as they appear in a model's forward method
    2. Organized by module_fqn --&gt; feature_name --&gt; feature values
  - Helps generate visualizations of three different types:
    - A formatted table
    - A line plot (for both per-tensor and per-channel statistics)
    - A histogram (for both per-tensor and per-channel statistics)
- <code>model_report.py</code>: File containing the <code>ModelReport</code> class
  - Main class users are interacting with to go through the ModelReport workflow
  - API described in detail in <a href="#modelreport-overview">Overview section</a></p>
<h1>Tests</h1>
<p>Tests for the ModelReport API are found in the <code>test_model_report_fx.py</code> file found <a href="https://github.com/pytorch/pytorch/blob/master/test/quantization/fx/test_model_report_fx.py">here</a>.</p>
<p>These tests include:
- Test class for the <code>ModelReportObserver</code>
- Test class for the <code>ModelReport</code> class
- Test class for the <code>ModelReportVisualizer</code> class
- Test class for <strong>each</strong> of the implemented Detectors</p>
<p>If you wish to add a Detector, make sure to create a test class modeled after one of the existing classes and test your detector.
Because users will be interacting with the Detectors through the <code>ModelReport</code> class and not directly, ensure that the tests follow this as well.</p>
<h1>Future Tasks and Improvements</h1>
<p>Below is a list of tasks that can help further improve the API or bug fixes that give the API more stability:</p>
<ul>
<li>[ ] For DynamicStaticDetector, change method of calculating stationarity from variance to variance of variance to help account for outliers</li>
<li>[ ] Add more types of visualizations for data</li>
<li>[ ] Add ability to visualize histograms of histogram observers</li>
<li>[ ] Automatically generate QConfigs from given suggestions</li>
<li>[ ] Tune default arguments for detectors with further research and analysis on what appropriate thresholds are</li>
<li>[ ] Merge the generation of the reports and the qconfig generation together</li>
<li>[ ] Make a lot of the dicts returned object classes</li>
<li>[ ] Change type of equalization config from <code>QConfigMapping</code> to <code>EqualizationMapping</code></li>
</ul>