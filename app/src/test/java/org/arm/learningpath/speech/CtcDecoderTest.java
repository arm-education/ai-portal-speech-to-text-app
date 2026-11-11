package org.arm.learningpath.speech;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public final class CtcDecoderTest {
    private static final int BLANK = 0;
    private static final int A = 1;
    private static final int B = 2;
    private static final int DELIMITER = 3;
    private static final String[] VOCABULARY = {"<pad>", "A", "B", "|"};

    private final CtcDecoder decoder = new CtcDecoder(VOCABULARY, BLANK, "|", false);

    @Test
    public void collapsesConsecutiveLabels() {
        String result = decoder.decode(logits(A, A, A), 3);

        assertEquals("A", result);
    }

    @Test
    public void preservesRepeatedLabelSeparatedByBlank() {
        String result = decoder.decode(logits(A, BLANK, A), 3);

        assertEquals("AA", result);
    }

    @Test
    public void convertsWordDelimiterToSpace() {
        String result = decoder.decode(logits(A, DELIMITER, DELIMITER, B), 4);

        assertEquals("A B", result);
    }

    @Test
    public void trimsLeadingAndTrailingDelimiters() {
        assertEquals("A", decoder.decode(logits(DELIMITER, A, DELIMITER), 3));
    }

    @Test
    public void ignoresPredictionsInPaddedTail() {
        String result = decoder.decode(logits(A, BLANK, B, DELIMITER, B), 2);

        assertEquals("A", result);
    }

    @Test
    public void doesNotReadInvalidScoresBeyondValidFrames() {
        float[] scores = logits(A, B);
        Arrays.fill(scores, VOCABULARY.length, scores.length, Float.NaN);

        assertEquals("A", decoder.decode(scores, 1));
    }

    @Test
    public void blankOrEmptyPredictionsProduceNoText() {
        String blanks = decoder.decode(logits(BLANK, BLANK), 2);
        assertEquals("", blanks);

        String empty = decoder.decode(new float[0], 0);
        assertEquals("", empty);
    }

    @Test
    public void appliesLowerCaseOnlyWhenConfigured() {
        CtcDecoder lowerCaseDecoder = new CtcDecoder(VOCABULARY, BLANK, "|", true);
        float[] scores = logits(A, B);

        assertEquals("ab", lowerCaseDecoder.decode(scores, 2));
        assertEquals("AB", decoder.decode(scores, 2));
    }

    @Test
    public void rejectsMalformedScoreDimensions() {
        assertThrows(IllegalArgumentException.class, () -> decoder.decode(null, 0));
        assertThrows(IllegalArgumentException.class, () -> decoder.decode(new float[3], 0));
        assertThrows(IllegalArgumentException.class, () -> decoder.decode(logits(A), -1));
        assertThrows(IllegalArgumentException.class, () -> decoder.decode(logits(A), 2));
    }

    @Test
    public void rejectsNonFiniteScoresEvenWhenTheyWouldNotWinArgmax() {
        for (float invalid : new float[]{Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
            float[] scores = logits(A);
            scores[B] = invalid;

            assertThrows(IllegalStateException.class, () -> decoder.decode(scores, 1));
        }
    }

    private static float[] logits(int... labels) {
        float[] scores = new float[labels.length * VOCABULARY.length];
        Arrays.fill(scores, -5.0f);
        for (int frame = 0; frame < labels.length; frame++) {
            scores[frame * VOCABULARY.length + labels[frame]] = 5.0f;
        }
        return scores;
    }
}
