package org.arm.learningpath.whisper.litert;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Minimal decoder for the byte-level BPE vocabulary stored in tokenizer.json. */
final class WhisperTokenizer {
    private final Map<Integer, String> idToToken = new HashMap<>();
    private final Set<Integer> specialIds = new HashSet<>();
    private final Map<Integer, Integer> unicodeToByte = buildUnicodeToByteMap();

    WhisperTokenizer(File tokenizerFile) throws IOException {
        if (tokenizerFile == null || !tokenizerFile.isFile()) {
            throw new IOException("The LiteRT package is missing tokenizer.json");
        }

        try {
            JSONObject root = new JSONObject(readUtf8(tokenizerFile));
            JSONObject vocabulary = root.getJSONObject("model").getJSONObject("vocab");
            Iterator<String> keys = vocabulary.keys();
            while (keys.hasNext()) {
                String token = keys.next();
                idToToken.put(vocabulary.getInt(token), token);
            }

            JSONArray addedTokens = root.optJSONArray("added_tokens");
            if (addedTokens != null) {
                for (int index = 0; index < addedTokens.length(); index++) {
                    JSONObject entry = addedTokens.getJSONObject(index);
                    int id = entry.getInt("id");
                    idToToken.put(id, entry.getString("content"));
                    if (entry.optBoolean("special", false)) {
                        specialIds.add(id);
                    }
                }
            }
        } catch (JSONException exception) {
            throw new IOException("tokenizer.json is not a compatible Whisper tokenizer", exception);
        }
    }

    String decode(List<Integer> tokenIds) {
        StringBuilder encoded = new StringBuilder();
        for (int id : tokenIds) {
            String token = idToToken.get(id);
            if (token == null) {
                throw new IllegalStateException("tokenizer.json has no entry for token " + id);
            }
            if (!specialIds.contains(id) && !looksSpecial(token)) {
                encoded.append(token);
            }
        }

        ByteArrayOutputStream bytes = new ByteArrayOutputStream(encoded.length());
        for (int offset = 0; offset < encoded.length();) {
            int codePoint = encoded.codePointAt(offset);
            Integer value = unicodeToByte.get(codePoint);
            if (value != null) {
                bytes.write(value);
            }
            offset += Character.charCount(codePoint);
        }
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8).trim();
    }

    private static boolean looksSpecial(String token) {
        return token.startsWith("<|") && token.endsWith("|>");
    }

    private static String readUtf8(File file) throws IOException {
        long length = file.length();
        if (length <= 0 || length > Integer.MAX_VALUE) {
            throw new IOException("Invalid tokenizer.json size");
        }
        byte[] bytes = new byte[(int) length];
        int offset = 0;
        try (FileInputStream input = new FileInputStream(file)) {
            while (offset < bytes.length) {
                int count = input.read(bytes, offset, bytes.length - offset);
                if (count < 0) {
                    break;
                }
                offset += count;
            }
        }
        if (offset != bytes.length) {
            throw new IOException("Could not read tokenizer.json");
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static Map<Integer, Integer> buildUnicodeToByteMap() {
        boolean[] direct = new boolean[256];
        for (int value = 33; value <= 126; value++) {
            direct[value] = true;
        }
        for (int value = 161; value <= 172; value++) {
            direct[value] = true;
        }
        for (int value = 174; value <= 255; value++) {
            direct[value] = true;
        }

        Map<Integer, Integer> inverse = new HashMap<>(256);
        int extra = 0;
        for (int value = 0; value <= 255; value++) {
            int codePoint = direct[value] ? value : 256 + extra++;
            inverse.put(codePoint, value);
        }
        return inverse;
    }
}
