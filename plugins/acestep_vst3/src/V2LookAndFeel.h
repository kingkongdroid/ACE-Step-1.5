#pragma once

#include <JuceHeader.h>

namespace acestep::vst3
{
class V2LookAndFeel final : public juce::LookAndFeel_V4
{
public:
    V2LookAndFeel();

    juce::Font getTextButtonFont(juce::TextButton& button, int buttonHeight) override;
    juce::Font getComboBoxFont(juce::ComboBox& box) override;
    void drawButtonBackground(juce::Graphics& g,
                              juce::Button& button,
                              const juce::Colour& backgroundColour,
                              bool isMouseOverButton,
                              bool isButtonDown) override;
    void drawComboBox(juce::Graphics& g,
                      int width,
                      int height,
                      bool isButtonDown,
                      int buttonX,
                      int buttonY,
                      int buttonW,
                      int buttonH,
                      juce::ComboBox& box) override;
    void drawTextEditorOutline(juce::Graphics& g,
                               int width,
                               int height,
                               juce::TextEditor& textEditor) override;
};
}  // namespace acestep::vst3
