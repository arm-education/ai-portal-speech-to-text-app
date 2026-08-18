package org.arm.learningpath.whisper.litert;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/** Computes the Whisper log-Mel input expected by the LiteRT encoder signature. */
final class WhisperFeatureExtractor {
    private static final int FFT_SIZE = 400;
    private static final int HOP_LENGTH = 160;
    private static final int FREQUENCY_BINS = FFT_SIZE / 2 + 1;
    private static final double SAMPLE_RATE = 16_000.0;
    private static final float LOG_FLOOR = 1.0e-10f;

    private static final float[] WINDOW = createPeriodicHannWindow();
    private static final Map<Integer, float[]> FILTER_CACHE = new HashMap<>();

    private WhisperFeatureExtractor() {
    }

    /**
     * Pads or truncates the 16 kHz waveform, then returns a native-order FLOAT32
     * buffer laid out as {@code [1, melBins, frames]}.
     */
    static synchronized ByteBuffer extract(float[] recorded, int melBins, int frames) {
        if (recorded == null || recorded.length == 0) {
            throw new IllegalArgumentException("Record some audio before transcribing");
        }
        if (melBins <= 0 || melBins > 256 || frames <= 0 || frames > 10_000) {
            throw new IllegalArgumentException("Unsupported Whisper feature dimensions");
        }

        int sampleCount = Math.multiplyExact(frames, HOP_LENGTH);
        float[] waveform = new float[sampleCount];
        int copied = Math.min(recorded.length, sampleCount);
        for (int i = 0; i < copied; i++) {
            float value = recorded[i];
            waveform[i] = Float.isFinite(value) ? value : 0.0f;
        }

        float[] features = new float[Math.multiplyExact(melBins, frames)];
        float[] filters = FILTER_CACHE.computeIfAbsent(
                melBins,
                WhisperFeatureExtractor::createMelFilters
        );
        float[] frameSamples = new float[FFT_SIZE];
        float[] power = new float[FREQUENCY_BINS];
        RealFft400 fft = new RealFft400();
        float globalMaximum = Float.NEGATIVE_INFINITY;

        for (int frame = 0; frame < frames; frame++) {
            int start = frame * HOP_LENGTH - FFT_SIZE / 2;
            for (int index = 0; index < FFT_SIZE; index++) {
                int source = reflectIndex(start + index, sampleCount);
                frameSamples[index] = waveform[source] * WINDOW[index];
            }
            fft.powerSpectrum(frameSamples, power);

            for (int mel = 0; mel < melBins; mel++) {
                float energy = 0.0f;
                int filterOffset = mel * FREQUENCY_BINS;
                for (int bin = 0; bin < FREQUENCY_BINS; bin++) {
                    energy += filters[filterOffset + bin] * power[bin];
                }
                float value = (float) Math.log10(Math.max(energy, LOG_FLOOR));
                features[mel * frames + frame] = value;
                globalMaximum = Math.max(globalMaximum, value);
            }
        }

        float floor = globalMaximum - 8.0f;
        ByteBuffer output = ByteBuffer.allocateDirect(
                Math.multiplyExact(features.length, Float.BYTES)
        ).order(ByteOrder.nativeOrder());
        for (float feature : features) {
            output.putFloat((Math.max(feature, floor) + 4.0f) / 4.0f);
        }
        output.rewind();
        return output;
    }

    private static int reflectIndex(int rawIndex, int length) {
        int index = rawIndex;
        while (index < 0 || index >= length) {
            index = index < 0 ? -index : 2 * length - 2 - index;
        }
        return index;
    }

    private static float[] createPeriodicHannWindow() {
        float[] window = new float[FFT_SIZE];
        for (int index = 0; index < FFT_SIZE; index++) {
            window[index] = (float) (
                    0.5 - 0.5 * Math.cos(2.0 * Math.PI * index / FFT_SIZE)
            );
        }
        return window;
    }

    /** Builds the Slaney-normalized Mel bank used by WhisperFeatureExtractor. */
    private static float[] createMelFilters(int melBins) {
        double minimumMel = hzToMel(0.0);
        double maximumMel = hzToMel(SAMPLE_RATE / 2.0);
        double[] frequencies = new double[melBins + 2];
        for (int index = 0; index < frequencies.length; index++) {
            double mel = minimumMel
                    + (maximumMel - minimumMel) * index / (melBins + 1.0);
            frequencies[index] = melToHz(mel);
        }

        float[] filters = new float[Math.multiplyExact(melBins, FREQUENCY_BINS)];
        for (int mel = 0; mel < melBins; mel++) {
            double lower = frequencies[mel];
            double center = frequencies[mel + 1];
            double upper = frequencies[mel + 2];
            double normalization = 2.0 / (upper - lower);
            for (int bin = 0; bin < FREQUENCY_BINS; bin++) {
                double frequency = (SAMPLE_RATE / 2.0) * bin / (FREQUENCY_BINS - 1.0);
                double weight;
                if (frequency < lower || frequency > upper) {
                    weight = 0.0;
                } else if (frequency <= center) {
                    weight = (frequency - lower) / (center - lower);
                } else {
                    weight = (upper - frequency) / (upper - center);
                }
                filters[mel * FREQUENCY_BINS + bin] =
                        (float) (Math.max(0.0, weight) * normalization);
            }
        }
        return filters;
    }

    private static double hzToMel(double frequency) {
        double linearScale = 200.0 / 3.0;
        if (frequency < 1_000.0) {
            return frequency / linearScale;
        }
        return 15.0 + Math.log(frequency / 1_000.0) / (Math.log(6.4) / 27.0);
    }

    private static double melToHz(double mel) {
        double linearScale = 200.0 / 3.0;
        if (mel < 15.0) {
            return mel * linearScale;
        }
        return 1_000.0 * Math.exp((Math.log(6.4) / 27.0) * (mel - 15.0));
    }

    /** Exact length-400 DFT implemented with a cached length-1024 Bluestein FFT. */
    private static final class RealFft400 {
        private static final int CONVOLUTION_SIZE = 1_024;

        private final float[] chirpCos = new float[FFT_SIZE];
        private final float[] chirpSin = new float[FFT_SIZE];
        private final float[] kernelReal = new float[CONVOLUTION_SIZE];
        private final float[] kernelImaginary = new float[CONVOLUTION_SIZE];
        private final float[] workReal = new float[CONVOLUTION_SIZE];
        private final float[] workImaginary = new float[CONVOLUTION_SIZE];

        RealFft400() {
            for (int index = 0; index < FFT_SIZE; index++) {
                double angle = Math.PI * (long) index * index / FFT_SIZE;
                chirpCos[index] = (float) Math.cos(angle);
                chirpSin[index] = (float) Math.sin(angle);
                kernelReal[index] = chirpCos[index];
                kernelImaginary[index] = chirpSin[index];
                if (index != 0) {
                    kernelReal[CONVOLUTION_SIZE - index] = chirpCos[index];
                    kernelImaginary[CONVOLUTION_SIZE - index] = chirpSin[index];
                }
            }
            transformRadixTwo(kernelReal, kernelImaginary, false);
        }

        void powerSpectrum(float[] input, float[] output) {
            Arrays.fill(workReal, 0.0f);
            Arrays.fill(workImaginary, 0.0f);
            for (int index = 0; index < FFT_SIZE; index++) {
                float value = input[index];
                workReal[index] = value * chirpCos[index];
                workImaginary[index] = -value * chirpSin[index];
            }

            transformRadixTwo(workReal, workImaginary, false);
            for (int index = 0; index < CONVOLUTION_SIZE; index++) {
                float real = workReal[index];
                float imaginary = workImaginary[index];
                workReal[index] = real * kernelReal[index]
                        - imaginary * kernelImaginary[index];
                workImaginary[index] = real * kernelImaginary[index]
                        + imaginary * kernelReal[index];
            }
            transformRadixTwo(workReal, workImaginary, true);

            for (int bin = 0; bin < FREQUENCY_BINS; bin++) {
                float real = workReal[bin] * chirpCos[bin]
                        + workImaginary[bin] * chirpSin[bin];
                float imaginary = workImaginary[bin] * chirpCos[bin]
                        - workReal[bin] * chirpSin[bin];
                output[bin] = real * real + imaginary * imaginary;
            }
        }

        private static void transformRadixTwo(
                float[] real,
                float[] imaginary,
                boolean inverse
        ) {
            int size = real.length;
            for (int i = 1, j = 0; i < size; i++) {
                int bit = size >> 1;
                while ((j & bit) != 0) {
                    j ^= bit;
                    bit >>= 1;
                }
                j ^= bit;
                if (i < j) {
                    float realValue = real[i];
                    real[i] = real[j];
                    real[j] = realValue;
                    float imaginaryValue = imaginary[i];
                    imaginary[i] = imaginary[j];
                    imaginary[j] = imaginaryValue;
                }
            }

            for (int length = 2; length <= size; length <<= 1) {
                double angle = 2.0 * Math.PI / length * (inverse ? 1.0 : -1.0);
                float stepReal = (float) Math.cos(angle);
                float stepImaginary = (float) Math.sin(angle);
                int half = length >> 1;
                for (int offset = 0; offset < size; offset += length) {
                    float twiddleReal = 1.0f;
                    float twiddleImaginary = 0.0f;
                    for (int index = 0; index < half; index++) {
                        int even = offset + index;
                        int odd = even + half;
                        float oddReal = real[odd] * twiddleReal
                                - imaginary[odd] * twiddleImaginary;
                        float oddImaginary = real[odd] * twiddleImaginary
                                + imaginary[odd] * twiddleReal;
                        real[odd] = real[even] - oddReal;
                        imaginary[odd] = imaginary[even] - oddImaginary;
                        real[even] += oddReal;
                        imaginary[even] += oddImaginary;

                        float nextReal = twiddleReal * stepReal
                                - twiddleImaginary * stepImaginary;
                        twiddleImaginary = twiddleReal * stepImaginary
                                + twiddleImaginary * stepReal;
                        twiddleReal = nextReal;
                    }
                }
            }

            if (inverse) {
                for (int index = 0; index < size; index++) {
                    real[index] /= size;
                    imaginary[index] /= size;
                }
            }
        }
    }
}
