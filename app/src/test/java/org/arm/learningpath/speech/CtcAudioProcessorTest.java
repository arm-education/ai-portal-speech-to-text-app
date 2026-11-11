package org.arm.learningpath.speech;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public final class CtcAudioProcessorTest {
    private static final int[] KERNELS = {10, 3, 3, 3, 3, 2, 2};
    private static final int[] STRIDES = {5, 2, 2, 2, 2, 2, 2};

    @Test
    public void normalizesRecordingBeforeAddingZeroPadding() {
        float[] audio = {1.0f, 3.0f};

        float[] input = CtcAudioProcessor.normalizeAndPad(audio, 5);

        assertArrayEquals(new float[]{-0.99999995f, 0.99999995f, 0, 0, 0}, input, 1e-6f);
        assertArrayEquals(new float[]{1.0f, 3.0f}, audio, 0.0f);
    }

    @Test
    public void stabilizesNormalizationOfVeryQuietAudio() {
        float[] input = CtcAudioProcessor.normalizeAndPad(new float[]{-0.0001f, 0.0001f}, 2);

        assertArrayEquals(new float[]{-0.30151135f, 0.30151135f}, input, 1e-6f);
    }

    @Test
    public void constantRecordingBecomesFiniteZeros() {
        float[] input = CtcAudioProcessor.normalizeAndPad(new float[]{2.5f, 2.5f, 2.5f}, 5);

        assertArrayEquals(new float[5], input, 0.0f);
    }

    @Test
    public void silenceRemainsZeroAfterPadding() {
        float[] input = CtcAudioProcessor.normalizeAndPad(new float[3], 5);

        assertArrayEquals(new float[5], input, 0.0f);
    }

    @Test
    public void rejectsNonFiniteAudio() {
        for (float invalid : new float[]{Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
            assertThrows(IllegalArgumentException.class,
                    () -> CtcAudioProcessor.normalizeAndPad(new float[]{0.1f, invalid}, 3));
        }
    }

    @Test
    public void rejectsEmptyOrOversizedRecordings() {
        assertThrows(IllegalArgumentException.class,
                () -> CtcAudioProcessor.normalizeAndPad(null, 5));
        assertThrows(IllegalArgumentException.class,
                () -> CtcAudioProcessor.normalizeAndPad(new float[0], 5));
        assertThrows(IllegalArgumentException.class,
                () -> CtcAudioProcessor.normalizeAndPad(new float[6], 5));
        assertThrows(IllegalArgumentException.class,
                () -> CtcAudioProcessor.normalizeAndPad(new float[1], -1));
    }

    @Test
    public void fullThirtySecondWindowProduces1499Frames() {
        assertEquals(1499, CtcAudioProcessor.outputFrames(480_000, KERNELS, STRIDES));
    }

    @Test
    public void receptiveFieldBoundaryRequires400Samples() {
        assertEquals(1, CtcAudioProcessor.outputFrames(400, KERNELS, STRIDES));
        assertEquals(0, CtcAudioProcessor.outputFrames(399, KERNELS, STRIDES));
        assertEquals(0, CtcAudioProcessor.outputFrames(0, KERNELS, STRIDES));
    }

    @Test
    public void sevenSecondRecordingProduces349ValidFrames() {
        assertEquals(349, CtcAudioProcessor.outputFrames(112_000, KERNELS, STRIDES));
    }

    @Test
    public void rejectsInvalidConvolutionConfiguration() {
        assertThrows(IllegalArgumentException.class,
                () -> CtcAudioProcessor.outputFrames(-1, KERNELS, STRIDES));
        assertThrows(IllegalArgumentException.class,
                () -> CtcAudioProcessor.outputFrames(400, new int[0], new int[0]));
        assertThrows(IllegalArgumentException.class,
                () -> CtcAudioProcessor.outputFrames(400, new int[]{10}, new int[]{5, 2}));
        assertThrows(IllegalArgumentException.class,
                () -> CtcAudioProcessor.outputFrames(400, new int[]{0}, new int[]{5}));
        assertThrows(IllegalArgumentException.class,
                () -> CtcAudioProcessor.outputFrames(400, new int[]{10}, new int[]{0}));
    }
}
