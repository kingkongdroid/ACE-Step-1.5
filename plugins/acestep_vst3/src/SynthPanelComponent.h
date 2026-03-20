#pragma once

#include <JuceHeader.h>

namespace acestep::vst3
{
class SynthPanelComponent final : public juce::Component
{
public:
    SynthPanelComponent();

    void paint(juce::Graphics& g) override;
    void resized() override;

    juce::TextEditor& backendUrlEditor() noexcept;
    juce::TextEditor& promptEditor() noexcept;
    juce::TextEditor& lyricsEditor() noexcept;
    juce::TextEditor& seedEditor() noexcept;
    juce::ComboBox& durationBox() noexcept;
    juce::ComboBox& modelBox() noexcept;
    juce::ComboBox& qualityBox() noexcept;

private:
    juce::Label backendUrlLabel_;
    juce::Label promptLabel_;
    juce::Label lyricsLabel_;
    juce::Label durationLabel_;
    juce::Label seedLabel_;
    juce::Label modelLabel_;
    juce::Label qualityLabel_;
    juce::TextEditor backendUrlEditor_;
    juce::TextEditor promptEditor_;
    juce::TextEditor lyricsEditor_;
    juce::TextEditor seedEditor_;
    juce::ComboBox durationBox_;
    juce::ComboBox modelBox_;
    juce::ComboBox qualityBox_;
};
}  // namespace acestep::vst3
