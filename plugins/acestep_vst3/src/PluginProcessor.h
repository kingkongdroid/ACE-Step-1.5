#pragma once

#include <JuceHeader.h>

#include "PluginConfig.h"
#include "PluginPreview.h"
#include "PluginState.h"

namespace acestep::vst3
{
class ACEStepVST3AudioProcessor final : public juce::AudioProcessor
{
public:
    ACEStepVST3AudioProcessor();
    ~ACEStepVST3AudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    bool isSynth() const;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    const PluginState& getState() const noexcept;
    PluginState& getMutableState() noexcept;
    [[nodiscard]] bool loadPreviewFile(const juce::File& file);
    void clearPreviewFile();
    void playPreview();
    void stopPreview();
    void revealPreviewFile() const;
    [[nodiscard]] bool hasPreviewFile() const;
    [[nodiscard]] bool isPreviewPlaying() const;

private:
    void syncPreviewFromState();

    PluginState state_;
    PluginPreview preview_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ACEStepVST3AudioProcessor)
};
}  // namespace acestep::vst3
