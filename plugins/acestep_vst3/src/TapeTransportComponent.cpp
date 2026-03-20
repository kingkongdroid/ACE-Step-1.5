#include "TapeTransportComponent.h"

#include "V2Chrome.h"

namespace acestep::vst3
{
TapeTransportComponent::TapeTransportComponent()
{
    backendLabel_.setColour(juce::Label::textColourId, v2::kLabelMuted);
    stateLabel_.setColour(juce::Label::textColourId, v2::kLabelPrimary);
    messageLabel_.setColour(juce::Label::textColourId, v2::kAccentMint);
    errorLabel_.setColour(juce::Label::textColourId, v2::kAccentRed);
    for (auto* label : {&backendLabel_, &stateLabel_, &messageLabel_, &errorLabel_})
    {
        label->setJustificationType(juce::Justification::centredLeft);
        addAndMakeVisible(*label);
    }
    addAndMakeVisible(generateButton_);
}

void TapeTransportComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    v2::drawModule(g, bounds, "Tape Transport", v2::statusColour(jobStatus_));

    auto top = bounds.removeFromTop(180).reduced(30, 38);
    auto leftReel = top.removeFromLeft(150).toFloat();
    auto rightReel = top.removeFromRight(150).toFloat();
    auto display = top.reduced(12, 22);
    v2::drawTapeReel(g, leftReel, v2::kAccentBlue, jobStatus_ == JobStatus::queuedOrRunning);
    v2::drawTapeReel(g, rightReel, v2::kAccentMint, jobStatus_ == JobStatus::queuedOrRunning);
    v2::drawDisplay(g, display.toNearestInt(), true);
    g.setColour(v2::kLabelPrimary);
    g.setFont(juce::Font(juce::FontOptions(26.0f, juce::Font::bold)));
    g.drawText(toString(jobStatus_).toUpperCase(), display.reduced(16.0f, 12.0f), juce::Justification::centred);

    auto lamp = juce::Rectangle<float>(18.0f, 18.0f)
                    .withCentre({static_cast<float>(bounds.getRight() - 38), 36.0f});
    v2::drawLamp(g, lamp, v2::statusColour(backendStatus_), backendStatus_ == BackendStatus::ready);
}

void TapeTransportComponent::resized()
{
    auto area = getLocalBounds().reduced(22);
    area.removeFromTop(186);
    backendLabel_.setBounds(area.removeFromTop(20));
    stateLabel_.setBounds(area.removeFromTop(26));
    area.removeFromTop(4);
    messageLabel_.setBounds(area.removeFromTop(22));
    area.removeFromTop(8);
    errorLabel_.setBounds(area.removeFromTop(44));
    area.removeFromTop(12);
    generateButton_.setBounds(area.removeFromTop(40).removeFromLeft(220));
}

juce::TextButton& TapeTransportComponent::generateButton() noexcept { return generateButton_; }

void TapeTransportComponent::setTransportState(BackendStatus backendStatus,
                                               JobStatus jobStatus,
                                               const juce::String& message,
                                               const juce::String& errorText)
{
    backendStatus_ = backendStatus;
    jobStatus_ = jobStatus;
    backendLabel_.setText("Backend // " + toString(backendStatus), juce::dontSendNotification);
    stateLabel_.setText("Transport // " + toString(jobStatus), juce::dontSendNotification);
    messageLabel_.setText(message.isEmpty() ? "Transport armed." : message, juce::dontSendNotification);
    errorLabel_.setText(errorText.isEmpty() ? "No critical warnings." : errorText, juce::dontSendNotification);
    const auto busy = jobStatus == JobStatus::submitting || jobStatus == JobStatus::queuedOrRunning;
    generateButton_.setEnabled(!busy);
    generateButton_.setButtonText(busy ? "Rendering..." : "Render");
    repaint();
}
}  // namespace acestep::vst3
