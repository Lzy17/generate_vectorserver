<h1>Audio "frontend" library for feature generation</h1>
<p>A feature generation library (also called frontend) that receives raw audio
input, and produces filter banks (a vector of values).</p>
<p>The raw audio input is expected to be 16-bit PCM features, with a configurable
sample rate. More specifically the audio signal goes through a pre-emphasis
filter (optionally); then gets sliced into (potentially overlapping) frames and
a window function is applied to each frame; afterwards, we do a Fourier
transform on each frame (or more specifically a Short-Time Fourier Transform)
and calculate the power spectrum; and subsequently compute the filter banks.</p>
<p>By default the library is configured with a set of defaults to perform the
different processing tasks. This takes place with the frontend_util.c function:</p>
<p><code>c++
void FrontendFillConfigWithDefaults(struct FrontendConfig* config)</code></p>
<p>A single invocation looks like:</p>
<p><code>c++
struct FrontendConfig frontend_config;
FrontendFillConfigWithDefaults(&amp;frontend_config);
int sample_rate = 16000;
FrontendPopulateState(&amp;frontend_config, &amp;frontend_state, sample_rate);
int16_t* audio_data = ;  // PCM audio samples at 16KHz.
size_t audio_size = ;  // Number of audio samples.
size_t num_samples_read;  // How many samples were processed.
struct FrontendOutput output =
    FrontendProcessSamples(
        &amp;frontend_state, audio_data, audio_size, &amp;num_samples_read);
for (i = 0; i &lt; output.size; ++i) {
  printf("%d ", output.values[i]);  // Print the feature vector.
}</code></p>
<p>Something to note in the above example is that the frontend consumes as many
samples needed from the audio data to produce a single feature vector (according
to the frontend configuration). If not enough samples were available to generate
a feature vector, the returned size will be 0 and the values pointer will be
<code>NULL</code>.</p>
<p>An example of how to use the frontend is provided in frontend_main.cc and its
binary frontend_main. This example, expects a path to a file containing <code>int16</code>
PCM features at a sample rate of 16KHz, and upon execution will printing out
the coefficients according to the frontend default configuration.</p>
<h2>Extra features</h2>
<p>Extra features of this frontend library include a noise reduction module, as
well as a gain control module.</p>
<p><strong>Noise cancellation</strong>. Removes stationary noise from each channel of the signal
using a low pass filter.</p>
<p><strong>Gain control</strong>. A novel automatic gain control based dynamic compression to
replace the widely used static (such as log or root) compression. Disabled
by default.</p>
<h2>Memory map</h2>
<p>The binary frontend_memmap_main shows a sample usage of how to avoid all the
initialization code in your application, by first running
"frontend_generate_memmap" to create a header/source file that uses a baked in
frontend state. This command could be automated as part of your build process,
or you can just use the output directly.</p>