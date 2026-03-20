#pragma once

#include <JuceHeader.h>

#include "PluginEnums.h"

namespace acestep::vst3
{
class TapeTransportComponent final : public juce::Component
{
public:
    TapeTransportComponent();

    void paint(juce::Graphics& g) override;
    void resized() override;

    juce::TextButton& generateButton() noexcept;
    void setTransportState(BackendStatus backendStatus,
                           JobStatus jobStatus,
                           const juce::String& message,
                           const juce::String& errorText);

private:
    juce::Label backendLabel_;
    juce::Label stateLabel_;
    juce::Label messageLabel_;
    juce::Label errorLabel_;
    juce::TextButton generateButton_ {"Render"};
    BackendStatus backendStatus_ = BackendStatus::offline;
    JobStatus jobStatus_ = JobStatus::idle;
};
}  // namespace acestep::vst3
