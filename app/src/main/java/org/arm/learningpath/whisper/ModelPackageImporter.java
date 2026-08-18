package org.arm.learningpath.whisper;

import android.content.Context;
import android.net.Uri;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/** Imports a selected model ZIP into private app storage without exposing partial installs. */
public final class ModelPackageImporter {
    private static final String TAG = "WhisperModelImport";
    private static final int BUFFER_SIZE = 1024 * 1024;
    private static final long MAX_ARCHIVE_BYTES = 4L * 1024L * 1024L * 1024L;
    private static final long MAX_EXTRACTED_FILE_BYTES = 3L * 1024L * 1024L * 1024L;
    private static final long MAX_PREPROCESSOR_BYTES = 256L * 1024L * 1024L;
    private static final long MAX_TOKENIZER_BYTES = 64L * 1024L * 1024L;
    private static final long MAX_METADATA_BYTES = 16L * 1024L * 1024L;
    private static final int MAX_ARCHIVE_ENTRIES = 10_000;
    private static final long TINY_MODEL_MAX_BYTES = 250L * 1024L * 1024L;

    private final Context context;
    private final File modelsDirectory;

    public ModelPackageImporter(Context context) {
        if (context == null) {
            throw new IllegalArgumentException("context must not be null");
        }
        Context applicationContext = context.getApplicationContext();
        this.context = applicationContext == null ? context : applicationContext;
        modelsDirectory = new File(this.context.getFilesDir(), "models");
    }

    public synchronized File importPackage(
            Uri archiveUri,
            WhisperModelDescriptor descriptor,
            ProgressListener progressListener
    ) throws IOException {
        if (archiveUri == null || descriptor == null) {
            throw new IllegalArgumentException("Select a model package first");
        }
        ProgressListener progress = progressListener == null
                ? ProgressListener.NONE
                : progressListener;
        ensureModelsDirectory();

        String operationId = UUID.randomUUID().toString();
        File localArchive = new File(
                context.getCacheDir(),
                "whisper-model-import-" + operationId + ".zip"
        );
        File staging = new File(modelsDirectory, ".import-" + operationId);
        if (!staging.mkdir()) {
            throw new IOException("Could not create the model import directory");
        }

        try {
            progress.onProgress("Copying the selected model package", 0.05f);
            copyArchive(archiveUri, localArchive);

            progress.onProgress("Checking the model package", 0.20f);
            try (ZipFile zip = new ZipFile(localArchive)) {
                ArchiveSelection selection = selectEntries(zip, descriptor);
                validateSelection(selection, descriptor);
                validateLiteRtIdentity(selection, descriptor);
                extractSelection(zip, selection, staging, progress);
                validateExecuTorchIdentity(selection, staging, descriptor);
            }

            validateInstalledFiles(staging, descriptor);
            progress.onProgress("Installing " + descriptor.displayName(), 0.92f);
            File installed = replaceAtomically(staging, modelDirectory(descriptor));
            progress.onProgress("Model ready", 1.0f);
            return installed;
        } catch (IOException | RuntimeException error) {
            deleteQuietly(staging);
            throw error;
        } finally {
            deleteQuietly(localArchive);
        }
    }

    public File modelDirectory(WhisperModelDescriptor descriptor) {
        if (descriptor == null || !descriptor.id().matches("[A-Za-z0-9._-]+")) {
            throw new IllegalArgumentException("Invalid model identifier");
        }
        return new File(modelsDirectory, storageKey(descriptor));
    }

    public boolean isInstalled(WhisperModelDescriptor descriptor) {
        File directory = modelDirectory(descriptor);
        try {
            validateInstalledFiles(directory, descriptor);
            return true;
        } catch (IOException ignored) {
            return false;
        }
    }

    private void copyArchive(Uri source, File destination) throws IOException {
        InputStream sourceStream = context.getContentResolver().openInputStream(source);
        if (sourceStream == null) {
            throw new IOException("Android could not open the selected archive");
        }
        try (InputStream input = new BufferedInputStream(sourceStream, BUFFER_SIZE);
             BufferedOutputStream output = new BufferedOutputStream(
                     new FileOutputStream(destination), BUFFER_SIZE)) {
            copyLimited(input, output, MAX_ARCHIVE_BYTES, "The selected ZIP is too large");
        }
    }

    private ArchiveSelection selectEntries(
            ZipFile zip,
            WhisperModelDescriptor descriptor
    ) throws IOException {
        ArchiveSelection selection = new ArchiveSelection();
        Enumeration<? extends ZipEntry> entries = zip.entries();
        int entryCount = 0;
        while (entries.hasMoreElements()) {
            ZipEntry entry = entries.nextElement();
            entryCount++;
            if (entryCount > MAX_ARCHIVE_ENTRIES) {
                throw new IOException("The ZIP contains too many files");
            }
            if (entry.isDirectory() || isIgnored(entry.getName())) {
                continue;
            }

            String path = normalize(entry.getName());
            String baseName = baseName(path);
            if (AdapterRegistry.EXECUTORCH_ID.equals(descriptor.adapterId())) {
                if (baseName.endsWith(".pte")
                        && !baseName.equals("whisper_preprocessor.pte")) {
                    int priority = (path.contains("optimized") ? 20 : 0)
                            + (baseName.equals("model.pte") ? 10 : 1);
                    selection.model = better(selection.model, entry, priority);
                } else if (baseName.equals("whisper_preprocessor.pte")) {
                    int priority = path.contains("optimized") ? 20 : 10;
                    selection.preprocessor = better(
                            selection.preprocessor,
                            entry,
                            priority
                    );
                } else if (baseName.equals("tokenizer.json")) {
                    int priority = path.contains("optimized") ? 20 : 10;
                    selection.tokenizer = better(selection.tokenizer, entry, priority);
                }
            } else if (AdapterRegistry.LITERT_ID.equals(descriptor.adapterId())) {
                if (baseName.endsWith(".tflite")) {
                    int priority;
                    if (baseName.contains("int8")) {
                        priority = 100;
                    } else if (baseName.contains("fp32") || baseName.contains("float")) {
                        priority = 0;
                    } else {
                        priority = 50;
                    }
                    selection.model = better(selection.model, entry, priority);
                } else if (baseName.equals("tokenizer.json")) {
                    selection.tokenizer = better(selection.tokenizer, entry, 10);
                }
            }

            if (isPreservedMetadata(baseName)) {
                Candidate current = selection.metadata.get(baseName);
                selection.metadata.put(baseName, better(current, entry, 10));
            }
        }
        return selection;
    }

    private static void validateSelection(
            ArchiveSelection selection,
            WhisperModelDescriptor descriptor
    ) throws IOException {
        List<String> missing = new ArrayList<>();
        if (selection.model == null || selection.model.priority <= 0) {
            missing.add(AdapterRegistry.LITERT_ID.equals(descriptor.adapterId())
                    ? "the INT8 .tflite model"
                    : "the optimized model.pte");
        }
        if (selection.tokenizer == null) {
            missing.add("tokenizer.json");
        }
        if (AdapterRegistry.EXECUTORCH_ID.equals(descriptor.adapterId())
                && selection.preprocessor == null) {
            missing.add("whisper_preprocessor.pte");
        }
        if (!missing.isEmpty()) {
            throw new IOException(
                    "This is not the package for " + descriptor.displayName()
                            + ". Missing " + String.join(", ", missing)
                            + ". Expected " + descriptor.packageHint() + "."
            );
        }
    }

    private static void validateLiteRtIdentity(
            ArchiveSelection selection,
            WhisperModelDescriptor descriptor
    ) throws IOException {
        if (!AdapterRegistry.LITERT_ID.equals(descriptor.adapterId())
                || selection.model == null) {
            return;
        }
        String modelPath = normalize(selection.model.entry.getName());
        String expectedFamily;
        if (descriptor.id().contains("large-v3")) {
            expectedFamily = "large-v3";
        } else if (descriptor.id().contains("medium")) {
            expectedFamily = "medium";
        } else {
            expectedFamily = "base";
        }
        boolean namesAWhisperFamily = modelPath.contains("whisper-base")
                || modelPath.contains("whisper-medium")
                || modelPath.contains("whisper-large-v3");
        if (namesAWhisperFamily && !modelPath.contains("whisper-" + expectedFamily)) {
            throw new IOException(
                    "The selected ZIP contains a different Whisper model. Expected "
                            + descriptor.packageHint() + "."
            );
        }
    }

    private static void validateExecuTorchIdentity(
            ArchiveSelection selection,
            File staging,
            WhisperModelDescriptor descriptor
    ) throws IOException {
        if (!AdapterRegistry.EXECUTORCH_ID.equals(descriptor.adapterId())
                || selection.model == null) {
            return;
        }
        boolean expectsTiny = descriptor.id().contains("tiny");
        String modelPath = normalize(selection.model.entry.getName());
        Boolean packageIsTiny = null;
        if (modelPath.contains("whisper_tiny") || modelPath.contains("whisper-tiny")) {
            packageIsTiny = true;
        } else if (modelPath.contains("whisper_small")
                || modelPath.contains("whisper-small")) {
            packageIsTiny = false;
        }

        File metadata = new File(staging, "metadata.yaml");
        if (packageIsTiny == null && metadata.isFile()) {
            String text = new String(
                    Files.readAllBytes(metadata.toPath()),
                    StandardCharsets.UTF_8
            ).toLowerCase(Locale.ROOT);
            if (text.contains("whisper-tiny") || text.contains("whisper_tiny")) {
                packageIsTiny = true;
            } else if (text.contains("whisper-small") || text.contains("whisper_small")) {
                packageIsTiny = false;
            }
        }
        if (packageIsTiny == null) {
            packageIsTiny = new File(staging, "model.pte").length() < TINY_MODEL_MAX_BYTES;
        }
        if (packageIsTiny != expectsTiny) {
            throw new IOException(
                    "The selected ZIP contains Whisper "
                            + (packageIsTiny ? "Tiny" : "Small")
                            + ", but " + descriptor.displayName() + " is selected."
            );
        }
    }

    private static void extractSelection(
            ZipFile zip,
            ArchiveSelection selection,
            File staging,
            ProgressListener progress
    ) throws IOException {
        boolean liteRt = selection.model.entry.getName()
                .toLowerCase(Locale.ROOT)
                .endsWith(".tflite");
        extract(
                zip,
                selection.model.entry,
                new File(staging, liteRt ? "model.tflite" : "model.pte"),
                MAX_EXTRACTED_FILE_BYTES
        );
        progress.onProgress("Extracting the model", 0.62f);

        if (selection.preprocessor != null) {
            extract(
                    zip,
                    selection.preprocessor.entry,
                    new File(staging, "whisper_preprocessor.pte"),
                    MAX_PREPROCESSOR_BYTES
            );
        }
        extract(
                zip,
                selection.tokenizer.entry,
                new File(staging, "tokenizer.json"),
                MAX_TOKENIZER_BYTES
        );
        progress.onProgress("Extracting tokenizer files", 0.82f);

        for (Map.Entry<String, Candidate> item : selection.metadata.entrySet()) {
            Candidate candidate = item.getValue();
            if (candidate != null) {
                extract(
                        zip,
                        candidate.entry,
                        new File(staging, item.getKey()),
                        MAX_METADATA_BYTES
                );
            }
        }
    }

    private static void extract(
            ZipFile zip,
            ZipEntry entry,
            File destination,
            long maximumBytes
    )
            throws IOException {
        long declaredSize = entry.getSize();
        if (declaredSize > maximumBytes) {
            throw new IOException("A file in the ZIP is too large: " + entry.getName());
        }
        try (InputStream input = new BufferedInputStream(
                zip.getInputStream(entry), BUFFER_SIZE);
             BufferedOutputStream output = new BufferedOutputStream(
                     new FileOutputStream(destination), BUFFER_SIZE)) {
            copyLimited(
                    input,
                    output,
                    maximumBytes,
                    "A file in the ZIP expands beyond the supported size"
            );
        }
    }

    private static void validateInstalledFiles(
            File directory,
            WhisperModelDescriptor descriptor
    ) throws IOException {
        if (directory == null || !directory.isDirectory()) {
            throw new IOException("The model directory is missing");
        }
        List<String> missing = new ArrayList<>();
        String modelName = AdapterRegistry.LITERT_ID.equals(descriptor.adapterId())
                ? "model.tflite"
                : "model.pte";
        if (!nonEmptyFile(directory, modelName)) {
            missing.add(modelName);
        }
        if (!nonEmptyFile(directory, "tokenizer.json")) {
            missing.add("tokenizer.json");
        }
        if (AdapterRegistry.EXECUTORCH_ID.equals(descriptor.adapterId())
                && !nonEmptyFile(directory, "whisper_preprocessor.pte")) {
            missing.add("whisper_preprocessor.pte");
        }
        if (!missing.isEmpty()) {
            throw new IOException("The installed package is missing " + String.join(", ", missing));
        }
    }

    private File replaceAtomically(File staging, File target) throws IOException {
        File backup = new File(
                modelsDirectory,
                ".backup-" + target.getName() + "-" + UUID.randomUUID()
        );
        boolean hadExisting = target.exists();
        if (hadExisting && !target.renameTo(backup)) {
            throw new IOException("Could not preserve the currently installed model");
        }
        if (!staging.renameTo(target)) {
            if (hadExisting && !backup.renameTo(target)) {
                throw new IOException(
                        "Installation failed and the previous model remains at "
                                + backup.getAbsolutePath()
                );
            }
            throw new IOException("Could not move the imported model into app storage");
        }
        deleteQuietly(backup);
        return target;
    }

    private void ensureModelsDirectory() throws IOException {
        if (!modelsDirectory.isDirectory() && !modelsDirectory.mkdirs()) {
            throw new IOException("Could not create the models directory");
        }
    }

    private static Candidate better(Candidate current, ZipEntry entry, int priority) {
        if (current == null || priority > current.priority
                || (priority == current.priority
                && normalize(entry.getName()).length()
                < normalize(current.entry.getName()).length())) {
            return new Candidate(entry, priority);
        }
        return current;
    }

    private static boolean isPreservedMetadata(String baseName) {
        return baseName.equals("processor_config.json")
                || baseName.equals("tokenizer_config.json")
                || baseName.equals("normalizer.json")
                || baseName.equals("metadata.yaml")
                || baseName.equals("config.yaml");
    }

    private static boolean isIgnored(String entryName) {
        String path = "/" + normalize(entryName);
        return path.contains("/.cache/")
                || path.contains("/.git/")
                || path.endsWith("/.git");
    }

    private static String normalize(String path) {
        return path.replace('\\', '/').toLowerCase(Locale.ROOT);
    }

    private static String baseName(String path) {
        int separator = path.lastIndexOf('/');
        return separator >= 0 ? path.substring(separator + 1) : path;
    }

    private static boolean nonEmptyFile(File directory, String name) {
        File file = new File(directory, name);
        return file.isFile() && file.length() > 0;
    }

    private static String storageKey(WhisperModelDescriptor descriptor) {
        // Keep the original V2 locations so an app upgrade can reuse imported models.
        if ("whisper-tiny-executorch-v2".equals(descriptor.id())) {
            return "tiny";
        }
        if ("whisper-small-executorch-v2".equals(descriptor.id())) {
            return "small";
        }
        return descriptor.id();
    }

    private static void copyLimited(
            InputStream input,
            BufferedOutputStream output,
            long maximumBytes,
            String limitMessage
    ) throws IOException {
        byte[] buffer = new byte[BUFFER_SIZE];
        long total = 0L;
        int read;
        while ((read = input.read(buffer)) != -1) {
            total += read;
            if (total > maximumBytes) {
                throw new IOException(limitMessage);
            }
            output.write(buffer, 0, read);
        }
    }

    private static void deleteQuietly(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) {
                deleteQuietly(child);
            }
        }
        if (!file.delete()) {
            Log.w(TAG, "Could not remove temporary file " + file);
        }
    }

    private static final class Candidate {
        final ZipEntry entry;
        final int priority;

        Candidate(ZipEntry entry, int priority) {
            this.entry = entry;
            this.priority = priority;
        }
    }

    private static final class ArchiveSelection {
        Candidate model;
        Candidate preprocessor;
        Candidate tokenizer;
        final Map<String, Candidate> metadata = new LinkedHashMap<>();
    }
}
