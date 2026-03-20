#include "ResultDeckComponent.h"

#include "V2Chrome.h"

namespace acestep::vst3
{
ResultDeckComponent::ResultDeckComponent()
{
    resultLabel_.setText("Take Select", juce::dontSendNotification);
    resultLabel_.setColour(juce::Label::textColourId, v2::kLabelMuted);
    summaryLabel_.setColour(juce::Label::textColourId, v2::kLabelPrimary);
    summaryLabel_.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(resultLabel_);
    addAndMakeVisible(resultSelector_);
    addAndMakeVisible(summaryLabel_);
}

void ResultDeckComponent::paint(juce::Graphics& g)
{
    v2::drawModule(g, getLocalBounds(), "Result Deck", v2::kAccentBlue);
}

void ResultDeckComponent::resized()
{
    auto area = getLocalBounds().reduced(18);
    area.removeFromTop(24);
    resultLabel_.setBounds(area.removeFromTop(18));
    resultSelector_.setBounds(area.removeFromTop(32));
    area.removeFromTop(10);
    summaryLabel_.setBounds(area.removeFromTop(52));
}

juce::ComboBox& ResultDeckComponent::resultSelector() noexcept { return resultSelector_; }

void ResultDeckComponent::setTakeSummary(const juce::String& title, const juce::String& detail)
{
    summaryLabel_.setText(title + "\n" + detail, juce::dontSendNotification);
}
}  // namespace acestep::vst3
