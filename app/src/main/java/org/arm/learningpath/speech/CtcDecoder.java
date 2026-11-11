package org.arm.learningpath.speech;

import java.util.Locale;

/** Greedy CTC decoding shared by compatible character-vocabulary exports. */
final class CtcDecoder {
    private final String[] vocabulary;
    private final int blankId;
    private final String wordDelimiter;
    private final boolean lowerCase;

    CtcDecoder(String[] vocabulary, int blankId, String wordDelimiter, boolean lowerCase) {
        if (vocabulary == null || blankId < 0 || blankId >= vocabulary.length
                || wordDelimiter == null || wordDelimiter.isEmpty()) {
            throw new IllegalArgumentException("Invalid CTC vocabulary configuration");
        }
        for (String token : vocabulary) {
            if (token == null || token.isEmpty()) {
                throw new IllegalArgumentException("The CTC vocabulary has a missing token");
            }
        }
        this.vocabulary = vocabulary.clone();
        this.blankId = blankId;
        this.wordDelimiter = wordDelimiter;
        this.lowerCase = lowerCase;
    }

    String decode(float[] logits, int validFrames) {
        if (logits == null || validFrames < 0
                || logits.length % vocabulary.length != 0
                || validFrames > logits.length / vocabulary.length) {
            throw new IllegalArgumentException("Unexpected CTC score dimensions");
        }
        StringBuilder text = new StringBuilder();
        int previous = -1;
        for (int frame = 0; frame < validFrames; frame++) {
            int offset = frame * vocabulary.length;
            int selected = 0;
            float best = Float.NEGATIVE_INFINITY;
            for (int token = 0; token < vocabulary.length; token++) {
                float score = logits[offset + token];
                if (!Float.isFinite(score)) {
                    throw new IllegalStateException("The CTC model returned an invalid score");
                }
                if (score > best) {
                    best = score;
                    selected = token;
                }
            }
            // Collapse consecutive labels BEFORE removing blanks: A, blank, A
            // must produce AA, whereas A, A must produce A.
            if (selected != previous && selected != blankId) {
                String token = vocabulary[selected];
                text.append(token.equals(wordDelimiter) ? " " : token);
            }
            previous = selected;
        }
        String transcript = text.toString().trim();
        if (lowerCase) {
            transcript = transcript.toLowerCase(Locale.ROOT);
        }
        return transcript;
    }
}
