<h1>Observers</h1>
<h2>Usage</h2>
<p>Observers are a small framework that allow users to attach code to the execution of SimpleNets and Operators.</p>
<p>An example of an Observer is the <code>TimeObserver</code>, used as follows:</p>
<h3>C++</h3>
<p><code>unique_ptr&lt;TimeObserver&lt;NetBase&gt;&gt; net_ob =
    make_unique&lt;TimeObserver&lt;NetBase&gt;&gt;(net.get());
auto* ob = net-&gt;AttachObserver(std::move(net_ob));
net-&gt;Run();
LOG(INFO) &lt;&lt; "av time children: " &lt;&lt; ob-&gt;average_time_children();
LOG(INFO) &lt;&lt; "av time: " &lt;&lt; ob-&gt;average_time();</code></p>
<h3>Python</h3>
<p>```
model.net.AttachObserver("TimeObserver")
ws.RunNet(model.net)
ob = model.net.GetObserver("TimeObserver")</p>
<p>print("av time children:", ob.average_time_children())
print("av time:", ob.average_time())
```</p>
<h3>Histogram Observer</h3>
<p>Creates a histogram for the values of weights and activations</p>
<p><code>model.net.AddObserver("HistogramObserver",
                      "histogram.txt", # filename
                      2014, # number of bins in histogram
                      32 # Dumping frequency
                      )
ws.RunNet(model.net)</code></p>
<p>This will generate a histogram for the activations and store it in histogram.txt</p>
<h2>Implementing An Observer</h2>
<p>To implement an observer you must inherit from <code>ObserverBase</code> and implement the <code>Start</code> and <code>Stop</code> functions.</p>
<p>Observers are instantiated with a <code>subject</code> of a generic type, such as a <code>Net</code> or <code>Operator</code>.  The observer framework is built to be generic enough to "observe" various other types, however.</p>