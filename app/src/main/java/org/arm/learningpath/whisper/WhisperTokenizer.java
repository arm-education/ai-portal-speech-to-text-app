package org.arm.learningpath.whisper;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/** Minimal decoder for the byte-level BPE tokens emitted by Whisper. */
final class WhisperTokenizer {
    private final Map<Integer, String> tokensById = new HashMap<>();
    private final Map<Integer, Integer> byteDecoder = createByteDecoder();

    WhisperTokenizer(File tokenizerFile) throws Exception {
        String json = new String(
                Files.readAllBytes(tokenizerFile.toPath()),
                StandardCharsets.UTF_8
        );
        JSONObject root = new JSONObject(json);
        JSONObject vocabulary = root.getJSONObject("model").getJSONObject("vocab");
        Iterator<String> keys = vocabulary.keys();
        while (keys.hasNext()) {
            String token = keys.next();
            tokensById.put(vocabulary.getInt(token), token);
        }

        JSONArray addedTokens = root.optJSONArray("added_tokens");
        if (addedTokens != null) {
            for (int index = 0; index < addedTokens.length(); index++) {
                JSONObject token = addedTokens.getJSONObject(index);
                tokensById.put(token.getInt("id"), token.getString("content"));
            }
        }
    }

    String decode(List<Integer> tokenIds) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (int tokenId : tokenIds) {
            String token = tokensById.get(tokenId);
            if (token == null || isSpecialToken(token)) {
                continue;
            }
            for (int offset = 0; offset < token.length();) {
                int codePoint = token.codePointAt(offset);
                Integer value = byteDecoder.get(codePoint);
                if (value != null) {
                    bytes.write(value);
                } else {
                    byte[] fallback = new String(Character.toChars(codePoint))
                            .getBytes(StandardCharsets.UTF_8);
                    bytes.write(fallback, 0, fallback.length);
                }
                offset += Character.charCount(codePoint);
            }
        }
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8).trim();
    }

    private static boolean isSpecialToken(String token) {
        return token.startsWith("<|") && token.endsWith("|>");
    }

    private static Map<Integer, Integer> createByteDecoder() {
        int[] bytes = new int[256];
        int count = 0;
        for (int value = 33; value <= 126; value++) {
            bytes[count++] = value;
        }
        for (int value = 161; value <= 172; value++) {
            bytes[count++] = value;
        }
        for (int value = 174; value <= 255; value++) {
            bytes[count++] = value;
        }

        int[] codePoints = new int[256];
        System.arraycopy(bytes, 0, codePoints, 0, count);
        int extra = 0;
        for (int value = 0; value < 256; value++) {
            boolean present = false;
            for (int index = 0; index < count; index++) {
                if (bytes[index] == value) {
                    present = true;
                    break;
                }
            }
            if (!present) {
                bytes[count] = value;
                codePoints[count] = 256 + extra;
                count++;
                extra++;
            }
        }

        Map<Integer, Integer> decoder = new HashMap<>();
        for (int index = 0; index < count; index++) {
            decoder.put(codePoints[index], bytes[index]);
        }
        return decoder;
    }
}
