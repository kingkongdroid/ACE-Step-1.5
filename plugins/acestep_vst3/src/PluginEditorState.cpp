#include "PluginEditor.h"

#include "PluginConfig.h"
#include "PluginEnums.h"
#include "PluginProcessor.h"

namespace acestep::vst3
{
void ACEStepVST3AudioProcessorEditor::configureLabels()
{
    refreshStatusViews();
}

void ACEStepVST3AudioProcessorEditor::configureEditors()
{
    auto& backendEditor = synthPanel_.backendUrlEditor();
    auto& promptEditor = synthPanel_.promptEditor();
    auto& lyricsEditor = synthPanel_.lyricsEditor();
    auto& seedEditor = synthPanel_.seedEditor();

    backendEditor.setTextToShowWhenEmpty(kDefaultBackendBaseUrl, juce::Colours::grey);
    backendEditor.onTextChange = [this] { persistTextFields(); };

    promptEditor.setTextToShowWhenEmpty("Describe the tape pass you want to print.",
                                        juce::Colours::grey);
    promptEditor.onTextChange = [this] { persistTextFields(); };

    lyricsEditor.setTextToShowWhenEmpty("Optional lyric sketch or arrangement notes.",
                                        juce::Colours::grey);
    lyricsEditor.onTextChange = [this] { persistTextFields(); };

    seedEditor.setInputRestrictions(10, "0123456789");
    seedEditor.onTextChange = [this] { persistTextFields(); };
}

void ACEStepVST3AudioProcessorEditor::configureSelectors()
{
    auto& durationBox = synthPanel_.durationBox();
    auto& modelBox = synthPanel_.modelBox();
    auto& qualityBox = synthPanel_.qualityBox();
    auto& resultSlotBox = resultDeck_.resultSelector();

    for (const auto duration : {10, 30, 60, 120})
    {
        durationBox.addItem(juce::String(duration) + " seconds", duration);
    }
    modelBox.addItem(toString(ModelPreset::turbo), 1);
    modelBox.addItem(toString(ModelPreset::standard), 2);
    modelBox.addItem(toString(ModelPreset::quality), 3);
    qualityBox.addItem(toString(QualityMode::fast), 1);
    qualityBox.addItem(toString(QualityMode::balanced), 2);
    qualityBox.addItem(toString(QualityMode::high), 3);

    durationBox.onChange = [this] { persistTextFields(); };
    modelBox.onChange = [this] { persistTextFields(); };
    qualityBox.onChange = [this] { persistTextFields(); };
    resultSlotBox.onChange = [this] {
        if (isSyncing_)
        {
            return;
        }
        processor_.selectResultSlot(juce::jmax(0, resultDeck_.resultSelector().getSelectedItemIndex()));
        refreshStatusViews();
    };
    transport_.generateButton().onClick = [this] {
        persistTextFields();
        processor_.requestGeneration();
        refreshStatusViews();
    };
    previewDeck_.loadButton().onClick = [this] { choosePreviewFile(); };
    previewDeck_.playButton().onClick = [this] { playPreviewFile(); };
    previewDeck_.stopButton().onClick = [this] { stopPreviewFile(); };
    previewDeck_.clearButton().onClick = [this] { clearPreviewFile(); };
    previewDeck_.revealButton().onClick = [this] { revealPreviewFile(); };
}

void ACEStepVST3AudioProcessorEditor::syncFromProcessor()
{
    const auto& state = processor_.getState();
    isSyncing_ = true;
    synthPanel_.backendUrlEditor().setText(state.backendBaseUrl, juce::dontSendNotification);
    synthPanel_.promptEditor().setText(state.prompt, juce::dontSendNotification);
    synthPanel_.lyricsEditor().setText(state.lyrics, juce::dontSendNotification);
    synthPanel_.seedEditor().setText(juce::String(state.seed), juce::dontSendNotification);
    synthPanel_.durationBox().setSelectedId(state.durationSeconds, juce::dontSendNotification);
    synthPanel_.modelBox().setSelectedId(static_cast<int>(state.modelPreset) + 1,
                                         juce::dontSendNotification);
    synthPanel_.qualityBox().setSelectedId(static_cast<int>(state.qualityMode) + 1,
                                           juce::dontSendNotification);
    isSyncing_ = false;
}

void ACEStepVST3AudioProcessorEditor::persistTextFields()
{
    if (isSyncing_)
    {
        return;
    }

    auto& state = processor_.getMutableState();
    auto& backendEditor = synthPanel_.backendUrlEditor();
    auto& promptEditor = synthPanel_.promptEditor();
    auto& lyricsEditor = synthPanel_.lyricsEditor();
    auto& seedEditor = synthPanel_.seedEditor();
    auto& durationBox = synthPanel_.durationBox();
    auto& modelBox = synthPanel_.modelBox();
    auto& qualityBox = synthPanel_.qualityBox();

    state.backendBaseUrl = backendEditor.getText().trim();
    if (state.backendBaseUrl.isEmpty())
    {
        state.backendBaseUrl = kDefaultBackendBaseUrl;
        backendEditor.setText(state.backendBaseUrl, juce::dontSendNotification);
    }

    state.prompt = promptEditor.getText();
    state.lyrics = lyricsEditor.getText();
    state.durationSeconds = durationBox.getSelectedId() == 0 ? kDefaultDurationSeconds
                                                             : durationBox.getSelectedId();
    state.seed = seedEditor.getText().getIntValue();
    if (state.seed <= 0)
    {
        state.seed = kDefaultSeed;
        seedEditor.setText(juce::String(state.seed), juce::dontSendNotification);
    }

    state.modelPreset = static_cast<ModelPreset>(juce::jmax(0, modelBox.getSelectedItemIndex()));
    state.qualityMode = static_cast<QualityMode>(juce::jmax(0, qualityBox.getSelectedItemIndex()));
    refreshStatusViews();
}

void ACEStepVST3AudioProcessorEditor::refreshResultSelector()
{
    const auto& state = processor_.getState();
    auto& resultSelector = resultDeck_.resultSelector();
    isSyncing_ = true;
    resultSelector.clear(juce::dontSendNotification);
    for (int index = 0; index < kResultSlotCount; ++index)
    {
        auto label = state.resultSlots[static_cast<size_t>(index)];
        if (label.isEmpty())
        {
            label = "Result " + juce::String(index + 1) + " - empty";
        }
        resultSelector.addItem(label, index + 1);
    }
    resultSelector.setSelectedId(state.selectedResultSlot + 1, juce::dontSendNotification);
    isSyncing_ = false;
}

void ACEStepVST3AudioProcessorEditor::refreshStatusViews()
{
    const auto& state = processor_.getState();
    const auto sessionName = state.prompt.trim().isEmpty() ? "UNTITLED SESSION"
                                                           : state.prompt.substring(0, 48).toUpperCase();
    statusStrip_.setSessionName("SESSION // " + sessionName);
    statusStrip_.setModeName("TEXT MODE");
    statusStrip_.setBackendStatus(state.backendStatus);

    auto transportMessage = state.progressText;
    if (transportMessage.isEmpty())
    {
        transportMessage = "Slot " + juce::String(state.selectedResultSlot + 1) + " armed.";
    }
    transport_.setTransportState(state.backendStatus,
                                 state.jobStatus,
                                 transportMessage,
                                 state.errorMessage);

    auto takeTitle = "Take " + juce::String(state.selectedResultSlot + 1);
    auto takeDetail = state.resultSlots[static_cast<size_t>(state.selectedResultSlot)];
    if (takeDetail.isEmpty())
    {
        takeDetail = "No printed result yet.";
    }
    const auto& localPath = state.resultLocalPaths[static_cast<size_t>(state.selectedResultSlot)];
    const auto& remoteUrl = state.resultFileUrls[static_cast<size_t>(state.selectedResultSlot)];
    if (!localPath.isEmpty())
    {
        takeDetail += "\nLOCAL // " + localPath;
    }
    else if (!remoteUrl.isEmpty())
    {
        takeDetail += "\nREMOTE // " + remoteUrl;
    }
    resultDeck_.setTakeSummary(takeTitle, takeDetail);

    auto previewText = state.previewDisplayName.isEmpty() ? "No preview file loaded."
                                                          : "Loaded // " + state.previewDisplayName;
    previewText += "\n";
    previewText += state.previewFilePath.isEmpty() ? "Choose a local or generated file to preview."
                                                   : state.previewFilePath;
    if (processor_.isPreviewPlaying())
    {
        previewText += "\nPlayback // active";
    }
    previewDeck_.setPreviewSummary(previewText);
}
}  // namespace acestep::vst3
