package org.arm.learningpath.speech;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public final class ExecuTorchWhisperAdapterTest {
    private static final int VOCABULARY_SIZE = 51865;
    // IDs from the supported Tiny package's multilingual tokenizer.json.
    private static final int SPACE = 220;
    private static final int ANNOTATION_OPENER = 542; // " ["
    private static final int ARS = 685; // "ars", incorrectly suppressed by the English list
    private static final int BLANK_TEXT = 37592; // "BLANK", ordinary text
    private static final int END_OF_TEXT = 50257;

    @Test
    public void allowsEndOfTextAfterTheFirstGeneratedToken() {
        assertEquals(END_OF_TEXT, ExecuTorchWhisperAdapter.argmax(
                rankedLogits(END_OF_TEXT, BLANK_TEXT), false));
    }

    @Test
    public void suppressesEndOfTextForTheFirstGeneratedToken() {
        assertEquals(BLANK_TEXT, ExecuTorchWhisperAdapter.argmax(
                rankedLogits(END_OF_TEXT, BLANK_TEXT), true));
    }

    @Test
    public void suppressesSpaceOnlyForTheFirstGeneratedToken() {
        float[] logits = rankedLogits(SPACE, BLANK_TEXT);

        assertEquals(BLANK_TEXT, ExecuTorchWhisperAdapter.argmax(logits, true));
        assertEquals(SPACE, ExecuTorchWhisperAdapter.argmax(logits, false));
    }

    @Test
    public void suppressesAnnotationOpenerWithoutSuppressingItsText() {
        assertEquals(BLANK_TEXT, ExecuTorchWhisperAdapter.argmax(
                rankedLogits(ANNOTATION_OPENER, BLANK_TEXT), false));
    }

    @Test
    public void allowsMultilingualTextSuppressedByTheEnglishTokenList() {
        assertEquals(ARS, ExecuTorchWhisperAdapter.argmax(
                rankedLogits(ARS, BLANK_TEXT), false));
    }

    @Test
    public void suppressesDecoderControlTokensDuringTextGeneration() {
        int[] controlTokens = {
                50258, // start of transcript
                50358, // translate
                50359, // transcribe
                50360, // start of language model
                50361, // start of previous context
                50362  // no captions
        };

        for (int token : controlTokens) {
            assertEquals("Control token " + token, BLANK_TEXT,
                    ExecuTorchWhisperAdapter.argmax(rankedLogits(token, BLANK_TEXT), false));
        }
    }

    private static float[] rankedLogits(int preferredToken, int fallbackToken) {
        float[] logits = new float[VOCABULARY_SIZE];
        Arrays.fill(logits, Float.NEGATIVE_INFINITY);
        logits[preferredToken] = 2.0f;
        logits[fallbackToken] = 1.0f;
        return logits;
    }
}
