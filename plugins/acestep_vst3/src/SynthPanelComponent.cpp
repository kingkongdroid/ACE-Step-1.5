#include "SynthPanelComponent.h"

#include "V2Chrome.h"

namespace acestep::vst3
{
SynthPanelComponent::SynthPanelComponent()
{
    for (auto* label : {&backendUrlLabel_, &promptLabel_, &lyricsLabel_, &durationLabel_, &seedLabel_,
                        &modelLabel_, &qualityLabel_})
    {
        label->setColour(juce::Label::textColourId, v2::kLabelMuted);
        addAndMakeVisible(*label);
    }

    backendUrlLabel_.setText("Signal Path", juce::dontSendNotification);
    promptLabel_.setText("Prompt", juce::dontSendNotification);
    lyricsLabel_.setText("Lyrics", juce::dontSendNotification);
    durationLabel_.setText("Duration", juce::dontSendNotification);
    seedLabel_.setText("Seed", juce::dontSendNotification);
    modelLabel_.setText("Engine", juce::dontSendNotification);
    qualityLabel_.setText("Quality", juce::dontSendNotification);

    promptEditor_.setMultiLine(true);
    lyricsEditor_.setMultiLine(true);

    for (auto* component : {static_cast<juce::Component*>(&backendUrlEditor_),
                            static_cast<juce::Component*>(&promptEditor_),
                            static_cast<juce::Component*>(&lyricsEditor_),
                            static_cast<juce::Component*>(&seedEditor_),
                            static_cast<juce::Component*>(&durationBox_),
                            static_cast<juce::Component*>(&modelBox_),
                            static_cast<juce::Component*>(&qualityBox_)})
    {
        addAndMakeVisible(*component);
    }
}

void SynthPanelComponent::paint(juce::Graphics& g)
{
    v2::drawModule(g, getLocalBounds(), "Compose / Engine", v2::kAccentBlue);
}

void SynthPanelComponent::resized()
{
    auto area = getLocalBounds().reduced(18);
    area.removeFromTop(24);
    auto left = area.removeFromLeft(area.getWidth() / 2 + 10);
    auto right = area;
    const auto labelHeight = 18;
    const auto fieldHeight = 32;

    backendUrlLabel_.setBounds(left.removeFromTop(labelHeight));
    backendUrlEditor_.setBounds(left.removeFromTop(fieldHeight));
    left.removeFromTop(8);
    promptLabel_.setBounds(left.removeFromTop(labelHeight));
    promptEditor_.setBounds(left.removeFromTop(72));
    left.removeFromTop(8);
    lyricsLabel_.setBounds(left.removeFromTop(labelHeight));
    lyricsEditor_.setBounds(left.removeFromTop(90));

    durationLabel_.setBounds(right.removeFromTop(labelHeight));
    durationBox_.setBounds(right.removeFromTop(fieldHeight));
    right.removeFromTop(8);
    seedLabel_.setBounds(right.removeFromTop(labelHeight));
    seedEditor_.setBounds(right.removeFromTop(fieldHeight));
    right.removeFromTop(8);
    modelLabel_.setBounds(right.removeFromTop(labelHeight));
    modelBox_.setBounds(right.removeFromTop(fieldHeight));
    right.removeFromTop(8);
    qualityLabel_.setBounds(right.removeFromTop(labelHeight));
    qualityBox_.setBounds(right.removeFromTop(fieldHeight));
}

juce::TextEditor& SynthPanelComponent::backendUrlEditor() noexcept { return backendUrlEditor_; }
juce::TextEditor& SynthPanelComponent::promptEditor() noexcept { return promptEditor_; }
juce::TextEditor& SynthPanelComponent::lyricsEditor() noexcept { return lyricsEditor_; }
juce::TextEditor& SynthPanelComponent::seedEditor() noexcept { return seedEditor_; }
juce::ComboBox& SynthPanelComponent::durationBox() noexcept { return durationBox_; }
juce::ComboBox& SynthPanelComponent::modelBox() noexcept { return modelBox_; }
juce::ComboBox& SynthPanelComponent::qualityBox() noexcept { return qualityBox_; }
}  // namespace acestep::vst3
