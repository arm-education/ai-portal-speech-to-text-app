package org.arm.learningpath.speech;

/** Waveform normalization and valid output lengths for convolutional CTC models. */
final class CtcAudioProcessor {
    private CtcAudioProcessor() {
    }

    static float[] normalizeAndPad(float[] audio, int inputSamples) {
        if (audio == null || audio.length == 0 || audio.length > inputSamples) {
            throw new IllegalArgumentException("The recording does not fit this model's audio window");
        }
        double sum = 0;
        for (float sample : audio) {
            if (!Float.isFinite(sample)) {
                throw new IllegalArgumentException("The recording contains an invalid audio sample");
            }
            sum += sample;
        }
        double mean = sum / audio.length;
        double variance = 0;
        for (float sample : audio) {
            double centered = sample - mean;
            variance += centered * centered;
        }
        // Wav2Vec2FeatureExtractor uses population variance + 1e-7. Padding
        // must stay zero and must not participate in the normalization.
        double scale = 1.0 / Math.sqrt(variance / audio.length + 1e-7);
        float[] input = new float[inputSamples];
        for (int index = 0; index < audio.length; index++) {
            input[index] = (float) ((audio[index] - mean) * scale);
        }
        return input;
    }

    static int outputFrames(int samples, int[] kernels, int[] strides) {
        if (samples < 0 || kernels.length != strides.length || kernels.length == 0) {
            throw new IllegalArgumentException("Invalid CTC convolution configuration");
        }
        int frames = samples;
        for (int index = 0; index < kernels.length; index++) {
            if (kernels[index] <= 0 || strides[index] <= 0) {
                throw new IllegalArgumentException("Invalid CTC convolution configuration");
            }
            if (frames < kernels[index]) {
                return 0;
            }
            frames = (frames - kernels[index]) / strides[index] + 1;
        }
        return frames;
    }
}
