#pragma once

#include <JuceHeader.h>

namespace acestep::vst3
{
class ResultDeckComponent final : public juce::Component
{
public:
    ResultDeckComponent();

    void paint(juce::Graphics& g) override;
    void resized() override;

    juce::ComboBox& resultSelector() noexcept;
    void setTakeSummary(const juce::String& title, const juce::String& detail);

private:
    juce::Label resultLabel_;
    juce::ComboBox resultSelector_;
    juce::Label summaryLabel_;
};
}  // namespace acestep::vst3
