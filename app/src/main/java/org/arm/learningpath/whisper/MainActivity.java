package org.arm.learningpath.whisper;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsetsController;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.PopupMenu;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int MODEL_PACKAGE_REQUEST = 2101;
    private static final int SAVE_TRANSCRIPT_REQUEST = 2102;
    private static final int MICROPHONE_PERMISSION_REQUEST = 2103;
    private static final int MENU_CLEAR_ENTRY = 2201;
    private static final long MAX_RECORDING_MILLIS = 30_000L;
    private static final String PREFERENCES_NAME = "whisper-journal";
    private static final String PREF_SELECTED_MODEL = "selected-model";
    private static final String LEGACY_TINY_MODEL_ID = "whisper-tiny-executorch-v2";
    private static final String LEGACY_SMALL_MODEL_ID = "whisper-small-executorch-v2";
    private static final String TINY_MODEL_ID = "whisper-tiny-int8-xnnpack-executorch";
    private static final String SMALL_MODEL_ID = "whisper-small-int8-xnnpack-executorch";

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler timerHandler = new Handler(Looper.getMainLooper());

    private AdapterRegistry adapterRegistry;
    private ModelPackageImporter modelImporter;
    private AudioRecorder audioRecorder;
    private SharedPreferences preferences;
    private List<WhisperModelDescriptor> models;
    private WhisperModelDescriptor selectedModel;
    private SpeechToTextAdapter activeAdapter;

    private ScrollView screen;
    private View hero;
    private View recordDock;
    private TextView transcriptView;
    private TextView transcriptMeta;
    private TextView timerView;
    private TextView recordLabel;
    private TextView statusView;
    private TextView modelSummary;
    private TextView modelStatus;
    private ImageButton recordButton;
    private ProgressBar progressBar;
    private View actionPanel;
    private View modelPanelHeader;
    private View modelPanelDetails;
    private ImageView modelChevron;
    private Spinner modelSpinner;
    private Button importModelButton;
    private ImageButton copyButton;
    private ImageButton saveButton;
    private ImageButton moreButton;

    private String transcript = "";
    private long recordingStartedMillis;
    private boolean recording;
    private boolean working;
    private boolean modelReady;
    private boolean modelPanelExpanded;
    private volatile boolean destroyed;

    private final Runnable timerTick = new Runnable() {
        @Override
        public void run() {
            if (!recording || destroyed) {
                return;
            }
            long elapsed = Math.min(
                    MAX_RECORDING_MILLIS,
                    SystemClock.elapsedRealtime() - recordingStartedMillis
            );
            timerView.setText(formatTimer(elapsed));
            timerHandler.postDelayed(this, 200L);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        bindViews();
        applyWindowInsets();
        applySystemBarAppearance();

        models = ModelRegistry.models();
        modelImporter = new ModelPackageImporter(getApplicationContext());
        audioRecorder = new AudioRecorder();
        preferences = getSharedPreferences(PREFERENCES_NAME, MODE_PRIVATE);
        adapterRegistry = new AdapterRegistry();

        configureModelSpinner();
        updateTranscriptViews();
        updateControls();
        screen.setFocusableInTouchMode(true);
        screen.post(() -> {
            if (!destroyed) {
                screen.requestFocus();
                screen.scrollTo(0, 0);
            }
        });
    }

    private void bindViews() {
        screen = findViewById(R.id.screen);
        hero = findViewById(R.id.hero);
        recordDock = findViewById(R.id.record_dock);
        transcriptView = findViewById(R.id.transcript_view);
        transcriptMeta = findViewById(R.id.transcript_meta);
        timerView = findViewById(R.id.timer_view);
        recordLabel = findViewById(R.id.record_label);
        statusView = findViewById(R.id.status_view);
        modelSummary = findViewById(R.id.model_summary);
        modelStatus = findViewById(R.id.model_status);
        recordButton = findViewById(R.id.record_button);
        progressBar = findViewById(R.id.progress_bar);
        actionPanel = findViewById(R.id.action_panel);
        modelPanelHeader = findViewById(R.id.model_panel_header);
        modelPanelDetails = findViewById(R.id.model_panel_details);
        modelChevron = findViewById(R.id.model_chevron);
        modelSpinner = findViewById(R.id.model_spinner);
        importModelButton = findViewById(R.id.import_model_button);
        copyButton = findViewById(R.id.copy_button);
        saveButton = findViewById(R.id.save_button);
        moreButton = findViewById(R.id.more_button);

        recordButton.setOnClickListener(view -> toggleRecording());
        modelPanelHeader.setOnClickListener(view -> setModelPanelExpanded(!modelPanelExpanded));
        importModelButton.setOnClickListener(view -> openModelPackage());
        copyButton.setOnClickListener(view -> copyTranscript());
        saveButton.setOnClickListener(view -> openSaveTranscript());
        moreButton.setOnClickListener(this::showEntryActions);
    }

    private void configureModelSpinner() {
        if (models.isEmpty()) {
            modelStatus.setText("No compatible speech models are registered.");
            statusView.setText("No speech model is available.");
            updateControls();
            return;
        }

        String savedModelId = preferences.getString(
                PREF_SELECTED_MODEL,
                models.get(0).id()
        );
        if (LEGACY_TINY_MODEL_ID.equals(savedModelId)) {
            savedModelId = TINY_MODEL_ID;
            preferences.edit().putString(PREF_SELECTED_MODEL, savedModelId).apply();
        } else if (LEGACY_SMALL_MODEL_ID.equals(savedModelId)) {
            savedModelId = SMALL_MODEL_ID;
            preferences.edit().putString(PREF_SELECTED_MODEL, savedModelId).apply();
        }
        int selectedIndex = 0;
        for (int index = 0; index < models.size(); index++) {
            if (models.get(index).id().equals(savedModelId)) {
                selectedIndex = index;
                break;
            }
        }
        selectedModel = models.get(selectedIndex);

        ArrayAdapter<WhisperModelDescriptor> spinnerAdapter = new ArrayAdapter<>(
                this,
                R.layout.spinner_model_item,
                models
        );
        spinnerAdapter.setDropDownViewResource(R.layout.spinner_model_dropdown);
        modelSpinner.setAdapter(spinnerAdapter);
        modelSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                WhisperModelDescriptor model = models.get(position);
                if (model.equals(selectedModel)) {
                    return;
                }
                SpeechToTextAdapter previousAdapter = activeAdapter;
                selectedModel = model;
                preferences.edit().putString(PREF_SELECTED_MODEL, model.id()).apply();
                modelReady = false;
                activeAdapter = null;
                loadSelectedModelIfInstalled(previousAdapter);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        modelSpinner.setSelection(selectedIndex, false);
        loadSelectedModelIfInstalled(null);
    }

    private void loadSelectedModelIfInstalled(SpeechToTextAdapter previousAdapter) {
        WhisperModelDescriptor model = selectedModel;
        if (model == null) {
            return;
        }
        boolean installed = modelImporter.isInstalled(model);
        updateModelLabels(false);
        if (previousAdapter == null && !installed) {
            modelReady = false;
            statusView.setText("Import a model package to begin recording.");
            setModelPanelExpanded(true);
            updateControls();
            return;
        }

        setWorking(true, false);
        if (installed) {
            statusView.setText("Preparing " + model.displayName() + "...");
            modelStatus.setText("Loading the installed model...");
        } else {
            statusView.setText("Switching models...");
            modelStatus.setText("Releasing the previous model...");
        }
        executor.execute(() -> {
            try {
                if (previousAdapter != null) {
                    previousAdapter.close();
                }
                if (!installed) {
                    postToUi(() -> {
                        if (!model.equals(selectedModel)) {
                            return;
                        }
                        setWorking(false, false);
                        updateModelLabels(false);
                        statusView.setText("Import a model package to begin recording.");
                        setModelPanelExpanded(true);
                    });
                    return;
                }
                SpeechToTextAdapter adapter = adapterRegistry.forModel(model);
                adapter.load(model, modelImporter.modelDirectory(model));
                postToUi(() -> {
                    if (!model.equals(selectedModel)) {
                        return;
                    }
                    activeAdapter = adapter;
                    modelReady = adapter.isLoaded();
                    setWorking(false, false);
                    updateModelLabels(true);
                    statusView.setText("Ready. Tap Record when you want to begin.");
                });
            } catch (Throwable error) {
                postToUi(() -> showModelError("Model load failed", error));
            }
        });
    }

    private void openModelPackage() {
        if (selectedModel == null || recording || working) {
            return;
        }
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/zip");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{
                "application/zip",
                "application/x-zip-compressed",
                "application/octet-stream"
        });
        startActivityForResult(intent, MODEL_PACKAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            return;
        }
        Uri uri = data.getData();
        if (requestCode == MODEL_PACKAGE_REQUEST) {
            try {
                getContentResolver().takePersistableUriPermission(
                        uri,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION
                );
            } catch (SecurityException ignored) {
            }
            importSelectedModel(uri);
        } else if (requestCode == SAVE_TRANSCRIPT_REQUEST) {
            saveTranscript(uri);
        }
    }

    private void importSelectedModel(Uri packageUri) {
        WhisperModelDescriptor model = selectedModel;
        if (model == null) {
            return;
        }
        setWorking(true, true);
        statusView.setText("Checking the model package...");
        modelStatus.setText("Importing " + model.displayName() + "...");
        executor.execute(() -> {
            SpeechToTextAdapter adapter = adapterRegistry.forModel(model);
            try {
                adapter.close();
                File directory = modelImporter.importPackage(
                        packageUri,
                        model,
                        (message, fraction) -> postToUi(() -> {
                            statusView.setText(message);
                            setProgress(fraction);
                        })
                );
                postToUi(() -> {
                    progressBar.setIndeterminate(true);
                    statusView.setText("Preparing " + model.displayName() + "...");
                    modelStatus.setText("Loading the imported model...");
                });
                adapter.load(model, directory);
                postToUi(() -> {
                    if (!model.equals(selectedModel)) {
                        return;
                    }
                    activeAdapter = adapter;
                    modelReady = adapter.isLoaded();
                    setWorking(false, false);
                    updateModelLabels(true);
                    statusView.setText("Model ready. Tap Record to begin your entry.");
                });
            } catch (Throwable error) {
                boolean restored = false;
                try {
                    if (modelImporter.isInstalled(model)) {
                        adapter.load(model, modelImporter.modelDirectory(model));
                        restored = adapter.isLoaded();
                    }
                } catch (Throwable ignored) {
                }
                boolean existingModelRestored = restored;
                postToUi(() -> {
                    if (!existingModelRestored) {
                        showModelError("Model import failed", error);
                        return;
                    }
                    activeAdapter = adapter;
                    modelReady = true;
                    setWorking(false, false);
                    updateModelLabels(true);
                    String message = "Model import failed, but the previous model is still ready: "
                            + safeMessage(error);
                    modelStatus.setText(message);
                    statusView.setText(message);
                    setModelPanelExpanded(true);
                });
            }
        });
    }

    private void toggleRecording() {
        if (recording) {
            stopRecording();
        } else if (!transcript.isEmpty()) {
            confirmStartNewRecording();
        } else {
            requestOrStartRecording();
        }
    }

    private void confirmStartNewRecording() {
        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle(R.string.record_new_entry_title)
                .setMessage(R.string.record_new_entry_message)
                .setNegativeButton(R.string.cancel, null)
                .setPositiveButton(
                        R.string.record_new_entry,
                        (ignoredDialog, ignoredWhich) -> requestOrStartRecording()
                )
                .create();
        dialog.setOnShowListener(ignored -> dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                .setTextColor(getColor(R.color.journal_error)));
        dialog.show();
    }

    private void requestOrStartRecording() {
        if (!modelReady || activeAdapter == null) {
            statusView.setText("Choose and import a model before recording.");
            setModelPanelExpanded(true);
            return;
        }
        if (working) {
            return;
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    MICROPHONE_PERMISSION_REQUEST
            );
            return;
        }
        startRecording();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            String[] permissions,
            int[] grantResults
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != MICROPHONE_PERMISSION_REQUEST) {
            return;
        }
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording();
        } else {
            statusView.setText("Microphone access is required to record a journal entry.");
        }
    }

    private void startRecording() {
        try {
            audioRecorder.start(audio16Khz -> postToUi(() -> handleMaximumDuration(audio16Khz)));
            transcript = "";
            transcriptView.setText("Listening...");
            transcriptView.setTextColor(getColor(R.color.journal_faint));
            transcriptMeta.setVisibility(View.GONE);
            actionPanel.setVisibility(View.GONE);
            recording = true;
            recordingStartedMillis = SystemClock.elapsedRealtime();
            timerView.setText(formatTimer(0));
            timerHandler.removeCallbacks(timerTick);
            timerHandler.post(timerTick);
            getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            statusView.setText("Listening... tap Stop when you are finished.");
            updateControls();
        } catch (Throwable error) {
            statusView.setText("Recording could not start: " + safeMessage(error));
        }
    }

    private void stopRecording() {
        if (!recording) {
            return;
        }
        recording = false;
        finishRecordingUi();
        setWorking(true, false);
        statusView.setText("Finishing your recording...");
        executor.execute(() -> {
            try {
                float[] audio16Khz = audioRecorder.stop();
                transcribeOnWorker(audio16Khz);
            } catch (Throwable error) {
                postToUi(() -> showTranscriptionError(error));
            }
        });
    }

    private void handleMaximumDuration(float[] audio16Khz) {
        if (!recording) {
            return;
        }
        recording = false;
        finishRecordingUi();
        setWorking(true, false);
        timerView.setText(formatTimer(MAX_RECORDING_MILLIS));
        statusView.setText("Thirty seconds captured. Creating your transcript...");
        executor.execute(() -> transcribeOnWorker(audio16Khz));
    }

    private void finishRecordingUi() {
        timerHandler.removeCallbacks(timerTick);
        getWindow().clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        updateControls();
    }

    private void transcribeOnWorker(float[] audio16Khz) {
        if (audio16Khz == null || audio16Khz.length == 0) {
            postToUi(() -> showTranscriptionError(
                    new IllegalStateException("The recording did not contain audio")
            ));
            return;
        }
        if (audio16Khz.length < AudioRecorder.SAMPLE_RATE / 2) {
            postToUi(this::showShortRecording);
            return;
        }
        if (AudioRecorder.peak(audio16Khz) < 0.002f) {
            postToUi(this::showSilentRecording);
            return;
        }
        SpeechToTextAdapter adapter = activeAdapter;
        if (adapter == null || !adapter.isLoaded()) {
            postToUi(() -> showTranscriptionError(
                    new IllegalStateException("The selected model is not loaded")
            ));
            return;
        }
        try {
            postToUi(() -> statusView.setText("Turning your recording into text..."));
            TranscriptionResult result = adapter.transcribe(
                    audio16Khz,
                    (message, fraction) -> postToUi(() -> {
                        statusView.setText(message);
                        setProgress(fraction);
                    })
            );
            postToUi(() -> showTranscript(result));
        } catch (Throwable error) {
            postToUi(() -> showTranscriptionError(error));
        }
    }

    private void showTranscript(TranscriptionResult result) {
        transcript = result.text().trim();
        setWorking(false, false);
        if (transcript.isEmpty()) {
            transcriptView.setText("No speech was detected. Try recording again a little closer to the microphone.");
            transcriptView.setTextColor(getColor(R.color.journal_muted));
            transcriptMeta.setVisibility(View.GONE);
            actionPanel.setVisibility(View.GONE);
            statusView.setText("No speech detected.");
            updateControls();
            return;
        }

        transcriptView.setText(transcript);
        transcriptView.setTextColor(getColor(R.color.journal_ink));
        int wordCount = countWords(transcript);
        String backend = result.backend().isEmpty() ? selectedModel.displayName() : result.backend();
        transcriptMeta.setText(String.format(
                Locale.US,
                "%d words  \u00B7  %.1f seconds  \u00B7  %s",
                wordCount,
                result.elapsedMillis() / 1000.0,
                backend
        ));
        transcriptMeta.setVisibility(View.VISIBLE);
        actionPanel.setVisibility(View.VISIBLE);
        statusView.setText("Entry ready.");
        updateControls();
    }

    private void showTranscriptionError(Throwable error) {
        setWorking(false, false);
        transcript = "";
        transcriptView.setText("Your recording could not be transcribed. You can try again without leaving this screen.");
        transcriptView.setTextColor(getColor(R.color.journal_muted));
        transcriptMeta.setVisibility(View.GONE);
        actionPanel.setVisibility(View.GONE);
        statusView.setText("Transcription failed: " + safeMessage(error));
        updateControls();
    }

    private void showSilentRecording() {
        setWorking(false, false);
        transcript = "";
        transcriptView.setText(
                "This recording was nearly silent. Check the microphone and try speaking a little closer."
        );
        transcriptView.setTextColor(getColor(R.color.journal_muted));
        transcriptMeta.setVisibility(View.GONE);
        actionPanel.setVisibility(View.GONE);
        statusView.setText("No clear speech detected.");
        updateControls();
    }

    private void showShortRecording() {
        setWorking(false, false);
        transcript = "";
        transcriptView.setText(
                "That recording was too short. Try again and speak for a little longer."
        );
        transcriptView.setTextColor(getColor(R.color.journal_muted));
        transcriptMeta.setVisibility(View.GONE);
        actionPanel.setVisibility(View.GONE);
        statusView.setText("Recording too short.");
        updateControls();
    }

    private void copyTranscript() {
        if (transcript.isEmpty()) {
            return;
        }
        ClipboardManager clipboard = getSystemService(ClipboardManager.class);
        if (clipboard != null) {
            clipboard.setPrimaryClip(ClipData.newPlainText("Whisper Journal entry", transcript));
            Toast.makeText(this, "Entry copied", Toast.LENGTH_SHORT).show();
        }
    }

    private void openSaveTranscript() {
        if (transcript.isEmpty()) {
            return;
        }
        String timestamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("text/plain");
        intent.putExtra(Intent.EXTRA_TITLE, "whisper-journal-" + timestamp + ".txt");
        startActivityForResult(intent, SAVE_TRANSCRIPT_REQUEST);
    }

    private void showEntryActions(View anchor) {
        if (transcript.isEmpty() || working || recording) {
            return;
        }
        PopupMenu popup = new PopupMenu(this, anchor);
        popup.getMenu().add(0, MENU_CLEAR_ENTRY, 0, R.string.clear_entry);
        popup.setOnMenuItemClickListener(item -> {
            if (item.getItemId() != MENU_CLEAR_ENTRY) {
                return false;
            }
            confirmClearTranscript();
            return true;
        });
        popup.show();
    }

    private void confirmClearTranscript() {
        if (transcript.isEmpty()) {
            return;
        }
        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle(R.string.clear_entry_title)
                .setMessage(R.string.clear_entry_message)
                .setNegativeButton(R.string.cancel, null)
                .setPositiveButton(
                        R.string.clear_entry,
                        (ignoredDialog, ignoredWhich) -> clearTranscript()
                )
                .create();
        dialog.setOnShowListener(ignored -> dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                .setTextColor(getColor(R.color.journal_error)));
        dialog.show();
    }

    private void saveTranscript(Uri outputUri) {
        String entry = transcript;
        if (entry.isEmpty()) {
            return;
        }
        setWorking(true, false);
        statusView.setText("Saving your entry...");
        executor.execute(() -> {
            try (OutputStream output = getContentResolver().openOutputStream(outputUri, "w")) {
                if (output == null) {
                    throw new IllegalStateException("The selected file could not be opened");
                }
                output.write(entry.getBytes(StandardCharsets.UTF_8));
                output.flush();
                postToUi(() -> {
                    setWorking(false, false);
                    statusView.setText("Entry saved.");
                });
            } catch (Throwable error) {
                postToUi(() -> {
                    setWorking(false, false);
                    statusView.setText("Save failed: " + safeMessage(error));
                });
            }
        });
    }

    private void clearTranscript() {
        transcript = "";
        updateTranscriptViews();
        statusView.setText(modelReady
                ? "Ready. Tap Record when you want to begin."
                : "Import a model package to begin recording.");
        updateControls();
    }

    private void updateTranscriptViews() {
        transcriptView.setText(R.string.transcript_placeholder);
        transcriptView.setTextColor(getColor(R.color.journal_faint));
        transcriptMeta.setText("");
        transcriptMeta.setVisibility(View.GONE);
        actionPanel.setVisibility(View.GONE);
    }

    private void setModelPanelExpanded(boolean expanded) {
        modelPanelExpanded = expanded;
        modelPanelDetails.setVisibility(expanded ? View.VISIBLE : View.GONE);
        modelChevron.setImageResource(
                expanded ? R.drawable.ic_chevron_up : R.drawable.ic_chevron_down
        );
        modelChevron.setContentDescription(
                expanded ? "Collapse model settings" : getString(R.string.model_chevron)
        );
    }

    private void updateModelLabels(boolean loaded) {
        WhisperModelDescriptor model = selectedModel;
        if (model == null) {
            return;
        }
        boolean installed = modelImporter.isInstalled(model);
        String state = loaded ? "Ready" : (installed ? "Installed" : "Not installed");
        modelSummary.setText(model.displayName() + "  \u00B7  " + state);
        if (loaded) {
            modelStatus.setText("Installed and ready for private on-device transcription.");
        } else if (installed) {
            modelStatus.setText("Installed. The runtime is preparing this model.");
        } else {
            modelStatus.setText("Import " + model.packageHint() + " to use this model.");
        }
    }

    private void showModelError(String prefix, Throwable error) {
        modelReady = false;
        activeAdapter = null;
        setWorking(false, false);
        updateModelLabels(false);
        String message = prefix + ": " + safeMessage(error);
        modelStatus.setText(message);
        statusView.setText(message);
        setModelPanelExpanded(true);
    }

    private void setWorking(boolean value, boolean determinate) {
        working = value;
        progressBar.setVisibility(value ? View.VISIBLE : View.GONE);
        progressBar.setIndeterminate(!determinate);
        if (determinate) {
            progressBar.setMax(100);
            progressBar.setProgress(0);
        }
        updateControls();
    }

    private void setProgress(float fraction) {
        progressBar.setIndeterminate(false);
        progressBar.setMax(100);
        progressBar.setProgress(Math.max(0, Math.min(100, Math.round(fraction * 100.0f))));
    }

    private void updateControls() {
        boolean canStart = modelReady && !working;
        recordButton.setEnabled(recording || canStart);
        recordButton.setAlpha(recordButton.isEnabled() ? 1.0f : 0.42f);
        recordButton.setBackgroundResource(
                recording ? R.drawable.bg_record_stop : R.drawable.bg_record_idle
        );
        recordButton.setImageResource(recording ? R.drawable.ic_stop : R.drawable.ic_mic);
        recordButton.setContentDescription(recording
                ? getString(R.string.stop_recording)
                : getString(R.string.record));
        recordLabel.setText(recording ? R.string.stop_recording : R.string.record);

        boolean controlsEnabled = !recording && !working;
        modelSpinner.setEnabled(controlsEnabled);
        importModelButton.setEnabled(controlsEnabled && selectedModel != null);
        importModelButton.setAlpha(importModelButton.isEnabled() ? 1.0f : 0.5f);
        boolean entryActionsEnabled = controlsEnabled && !transcript.isEmpty();
        copyButton.setEnabled(entryActionsEnabled);
        saveButton.setEnabled(entryActionsEnabled);
        moreButton.setEnabled(entryActionsEnabled);
        float entryActionsAlpha = entryActionsEnabled ? 1.0f : 0.45f;
        copyButton.setAlpha(entryActionsAlpha);
        saveButton.setAlpha(entryActionsAlpha);
        moreButton.setAlpha(entryActionsAlpha);
    }

    private void applyWindowInsets() {
        int heroStart = hero.getPaddingStart();
        int heroTop = hero.getPaddingTop();
        int heroEnd = hero.getPaddingEnd();
        int heroBottom = hero.getPaddingBottom();
        int screenStart = screen.getPaddingStart();
        int screenTop = screen.getPaddingTop();
        int screenEnd = screen.getPaddingEnd();
        int screenBottom = screen.getPaddingBottom();
        int dockBottomMargin = ((ViewGroup.MarginLayoutParams)
                recordDock.getLayoutParams()).bottomMargin;
        screen.setOnApplyWindowInsetsListener((view, insets) -> {
            int topInset = Math.max(insets.getSystemWindowInsetTop(), insets.getStableInsetTop());
            int bottomInset = Math.max(
                    insets.getSystemWindowInsetBottom(),
                    insets.getStableInsetBottom()
            );
            hero.setPaddingRelative(
                    heroStart,
                    heroTop + topInset,
                    heroEnd,
                    heroBottom
            );
            screen.setPaddingRelative(
                    screenStart,
                    screenTop,
                    screenEnd,
                    screenBottom + bottomInset
            );
            ViewGroup.MarginLayoutParams dockParams = (ViewGroup.MarginLayoutParams)
                    recordDock.getLayoutParams();
            dockParams.bottomMargin = dockBottomMargin + bottomInset;
            recordDock.setLayoutParams(dockParams);
            return insets;
        });
        screen.requestApplyInsets();
    }

    @SuppressWarnings("deprecation")
    private void applySystemBarAppearance() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            WindowInsetsController controller = getWindow().getInsetsController();
            if (controller != null) {
                controller.setSystemBarsAppearance(
                        WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS,
                        WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS
                                | WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS
                );
            }
            return;
        }
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR
        );
    }

    private String formatTimer(long elapsedMillis) {
        long totalSeconds = elapsedMillis / 1000L;
        return String.format(
                Locale.US,
                "%02d:%02d / 00:30",
                totalSeconds / 60L,
                totalSeconds % 60L
        );
    }

    private static int countWords(String text) {
        String trimmed = text.trim();
        return trimmed.isEmpty() ? 0 : trimmed.split("\\s+").length;
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
                ? error.getClass().getSimpleName()
                : message;
    }

    private void postToUi(Runnable action) {
        runOnUiThread(() -> {
            if (!destroyed) {
                action.run();
            }
        });
    }

    @Override
    protected void onStop() {
        if (recording) {
            stopRecording();
        }
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        timerHandler.removeCallbacksAndMessages(null);
        AudioRecorder recorder = audioRecorder;
        AdapterRegistry registry = adapterRegistry;
        executor.execute(() -> {
            if (recorder != null) {
                try {
                    if (recorder.isRecording()) {
                        recorder.stop();
                    }
                } catch (Throwable ignored) {
                }
                recorder.release();
            }
            if (registry != null) {
                registry.close();
            }
        });
        executor.shutdown();
        super.onDestroy();
    }
}
