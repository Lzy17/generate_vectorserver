<h1>TensorFlow Lite Python image classification demo</h1>
<p>This <code>label_image.py</code> script shows how you can load a pre-trained and converted
TensorFlow Lite model and use it to recognize objects in images. The Python
script accepts arguments specifying the model to use, the corresponding labels
file, and the image to process.</p>
<p><strong>Tip:</strong> If you're using a Raspberry Pi, instead try the
<a href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi">classify_picamera.py example</a>.</p>
<p>Before you begin, make sure you
<a href="https://www.tensorflow.org/install">have TensorFlow installed</a>.</p>
<h2>Download sample model and image</h2>
<p>You can use any compatible model, but the following MobileNet v1 model offers a
good demonstration of a model trained to recognize 1,000 different objects.</p>
<p>```sh</p>
<h1>Get photo</h1>
<p>curl https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp &gt; /tmp/grace_hopper.bmp</p>
<h1>Get model</h1>
<p>curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz | tar xzv -C /tmp</p>
<h1>Get labels</h1>
<p>curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz  | tar xzv -C /tmp  mobilenet_v1_1.0_224/labels.txt</p>
<p>mv /tmp/mobilenet_v1_1.0_224/labels.txt /tmp/
```</p>
<h2>Run the sample</h2>
<p><code>sh
python3 label_image.py \
  --model_file /tmp/mobilenet_v1_1.0_224.tflite \
  --label_file /tmp/labels.txt \
  --image /tmp/grace_hopper.bmp</code></p>
<p>You should see results like this:</p>
<p><code>0.728693: military uniform
0.116163: Windsor tie
0.035517: bow tie
0.014874: mortarboard
0.011758: bolo tie</code></p>