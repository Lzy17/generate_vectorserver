<h1>TensorFlow Saved Model C API</h1>
<h2>Small ConcreteFunction Example</h2>
<p>The following example loads a saved model from <code>"/path/to/model"</code> and
executes a function <code>f</code> taking no arguments and returning one single
value (error checking is omitted for simplicity):</p>
<p>```c
TF_Status<em> status = TF_NewStatus();
TFE_ContextOptions</em> ctx_options = TFE_NewContextOptions();
TFE_Context* ctx = TFE_NewContext(ctx_options, status);</p>
<p>TF_SavedModel<em> saved_model = TF_LoadSavedModel("/path/to/model", ctx, status);
TF_ConcreteFunction</em> f = TF_GetSavedModelConcreteFunction(saved_model, "f", status);
TFE_Op* op = TF_ConcreteFunctionMakeCallOp(f, NULL, 0, status);</p>
<p>TFE_TensorHandle* output;
int nouts = 1;
TFE_Execute(op, &amp;output, &amp;nouts, status);</p>
<p>TFE_DeleteTensorHandle(output);
TFE_DeleteOp(op);
TFE_DeleteSavedModel(saved_model);
TFE_DeleteContext(ctx);
TFE_DeleteContextOptions(ctx_options);
TF_DeleteStatus(status);
```</p>