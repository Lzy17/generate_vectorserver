<h1>TensorFlow Spectrogram Example</h1>
<p>This example shows how you can load audio from a .wav file, convert it to a
spectrogram, and then save it out as a PNG image. A spectrogram is a
visualization of the frequencies in sound over time, and can be useful as a
feature for neural network recognition on noise or speech.</p>
<h2>Building</h2>
<p>To build it, run this command:</p>
<p><code>bash
bazel build tensorflow/examples/wav_to_spectrogram/...</code></p>
<p>That should build a binary executable that you can then run like this:</p>
<p><code>bash
bazel-bin/tensorflow/examples/wav_to_spectrogram/wav_to_spectrogram</code></p>
<p>This uses a default test audio file that's part of the TensorFlow source code,
and writes out the image to the current directory as spectrogram.png.</p>
<h2>Options</h2>
<p>To load your own audio, you need to supply a .wav file in LIN16 format, and use
the <code>--input_audio</code> flag to pass in the path.</p>
<p>To control how the spectrogram is created, you can specify the <code>--window_size</code>
and <code>--stride</code> arguments, which control how wide the window used to estimate
frequencies is, and how widely adjacent windows are spaced.</p>
<p>The <code>--output_image</code> flag sets the path to save the image file to. This is
always written out in PNG format, even if you specify a different file
extension.</p>
<p>If your result seems too dark, try using the <code>--brightness</code> flag to make the
output image easier to see.</p>
<p>Here's an example of how to use all of them together:</p>
<p><code>bash
bazel-bin/tensorflow/examples/wav_to_spectrogram/wav_to_spectrogram \
--input_wav=/tmp/my_audio.wav \
--window=1024 \
--stride=512 \
--output_image=/tmp/my_spectrogram.png</code></p>