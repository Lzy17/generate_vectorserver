<h1>TFLite Model Task Evaluation</h1>
<p>This page describes how you can check the accuracy of quantized models to verify
that any degradation in accuracy is within acceptable limits.</p>
<h2>Accuracy &amp; correctness</h2>
<p>TensorFlow Lite has two types of tooling to measure how accurately a delegate
behaves for a given model: Task-Based and Task-Agnostic.</p>
<p><strong>Task-Based Evaluation</strong> TFLite has two tools to evaluate correctness on two
image-based tasks: - <a href="http://image-net.org/challenges/LSVRC/2012/">ILSVRC 2012</a>
(Image Classification) with top-K accuracy -
<a href="https://cocodataset.org/#detection-2020">COCO Object Detection</a> (w/ bounding
boxes) with mean Average Precision (mAP)</p>
<p><strong>Task-Agnostic Evaluation</strong> For tasks where there isn't an established
on-device evaluation tool, or if you are experimenting with custom models,
TensorFlow Lite has the Inference Diff tool.</p>
<h2>Tools</h2>
<p>There are three different binaries which are supported. A brief description of
each is provided below.</p>
<h3><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/inference_diff#inference-diff-tool">Inference Diff Tool</a></h3>
<p>This binary compares TensorFlow Lite execution in single-threaded CPU inference
and user-defined inference.</p>
<h3><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification#image-classification-evaluation-based-on-ilsvrc-2012-task">Image Classification Evaluation</a></h3>
<p>This binary evaluates TensorFlow Lite models trained for the
<a href="http://www.image-net.org/challenges/LSVRC/2012/">ILSVRC 2012 image classification task.</a></p>
<h3><a href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/coco_object_detection#object-detection-evaluation-using-the-2014-coco-minival-dataset">Object Detection Evaluation</a></h3>
<p>This binary evaluates TensorFlow Lite models trained for the bounding box-based
<a href="https://cocodataset.org/#detection-eval">COCO Object Detection</a> task.</p>
<hr />
<p>For more information visit the TensorFlow Lite guide on
<a href="https://www.tensorflow.org/lite/performance/delegates#accuracy_correctness">Accuracy &amp; correctness</a>
page.</p>