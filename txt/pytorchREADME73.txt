<h1>Data Scheduler</h1>
<h2>Intro</h2>
<p>The data scheduler is used to control the update of the data sparsification parameters and works specifically with the data sparsifier class.
This class controls a specific config param (specified by the <code>schedule_param</code> argument) of
the data sparsifier class and varies it across the training process (or across time).</p>
<h2>API details</h2>
<p><code>BaseDataScheduler</code>: base class with abstract method <code>get_schedule_param</code> that computes the data sparsification parameter for all the data. The constructor accepts
1. <code>data_sparsifier</code>: The data sparsifier object whose parameter will be scheduled.
2. <code>schedule_param</code> : a specific config of the passed data sparsifier that needs to be scheduled/varied.</p>
<p><code>get_last_param</code>: gets the last scheduled parameter. Basically, a dictionary of name (of data) to schedule_param value mapping.</p>
<p><code>step</code>: Applies the <code>get_schedule_param</code> logic every epoch/step depending on when it is called. This should always be called after the <code>sparsifier.step()</code> has been called.</p>
<h2>Write your own data scheduler</h2>
<p>The custom data scheduler must be inherit from the <code>BaseDataScheduler</code> class and should have the <code>get_schedule_param()</code> function implemented. For example, that gradually multiplies the sparsity level by <code>gamma</code> every epoch.
It also takes an argument <code>threshold_sl</code> which when reached does not increase further.</p>
<p>```
class GammaScheduler(BaseDataScheduler):
    def <strong>init</strong>(self, data_sparsifier, gamma, threshold_sl):
        super().<strong>init</strong>(data_sparsifier, "sparsity_level")
        self.gamma = gamma
        self.threshold_sl = threshold_sl</p>
<pre><code>def get_schedule_param(self):
    if self.last_epoch &gt; 0:
        return {name: min(self.threshold_sl, config["sparsity_level"] * self.gamma) for name, config in self.data_sparsifier.data_groups.items()}
    else:
        return {name: 0.0 for name, config in self.data_sparsifier.data_groups.items()}
</code></pre>
<p>```</p>
<h2>Using data scheduler with data sparsifier</h2>
<p>Suppose the need is to vary data sparsity levels (or any sparsity <code>param</code>) during training, then a custom data scheduler can be implemented and used along with the data sparsifier.</p>
<p>Example:</p>
<p>```
model = SomeModel()
optimizer = SomeOptimizer(model.parameters(), lr=...)
data_sparsifier = SomeDataSparsifier(...)</p>
<p>data_scheduler = SomeDataScheduler(data_sparsifier, ...)</p>
<p>data_name = 'train_data'</p>
<p>for epoch in range(EPOCHS):
    for input, target in dataset:
        input = data_sparsifier.add_data(name=data_name, data=input)</p>
<pre><code>    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    data_sparsifier.step()

data_scheduler.step()
</code></pre>
<p>```</p>
<h3>Note:</h3>
<ol>
<li><code>get_schedule_param()</code> should return a dictionary wherein the keys are the names of the data and the values are the corresponding values of the <code>schedule_param</code> for the next step.</li>
<li>It is the responsibility of the <code>BaseDataScheduler</code> to call the <code>get_schedule_param()</code> when necessary.</li>
</ol>