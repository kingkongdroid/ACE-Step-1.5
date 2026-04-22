"""Unit tests for inference mode tensor handling in trainer modules."""

import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn

from acestep.training.trainer import PreprocessedLoRAModule, PreprocessedLoKRModule
from acestep.training.configs import LoRAConfig, LoKRConfig, TrainingConfig

class MockDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)

    def forward(self, hidden_states, **kwargs):
        return (self.q_proj(hidden_states),)

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = MockDecoder()
        self.config = MagicMock()
        self.null_condition_emb = nn.Parameter(torch.randn(1, 1, 128))

class TestInferenceModeHandling(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        # Put model in inference mode
        for param in self.model.parameters():
            param.data = param.data.to(device="cpu")
            # We simulate inference mode by using torch.no_grad() and cloning
            # but actual 'is_inference()' is hard to mock without real inference tensors
            # which are created by torch.inference_mode().

    @patch("acestep.training.trainer.check_peft_available", return_value=True)
    @patch("acestep.training.trainer.inject_lora_into_dit")
    def test_lora_module_handles_inference_tensors(self, mock_inject, mock_peft):
        mock_inject.return_value = (self.model, {"trainable_params": 100})

        lora_config = LoRAConfig()
        training_config = TrainingConfig(output_dir="/tmp/out")

        # Create an inference mode tensor for one parameter
        with torch.inference_mode():
            inf_param = torch.randn(128, 128)
            self.model.decoder.q_proj.weight.data = inf_param

        self.assertTrue(self.model.decoder.q_proj.weight.is_inference())

        # This should not raise "inference mode tensors do not allow storage access"
        module = PreprocessedLoRAModule(
            model=self.model,
            lora_config=lora_config,
            training_config=training_config,
            device="cpu",
            dtype=torch.float32
        )

        self.assertFalse(self.model.decoder.q_proj.weight.is_inference())
        mock_inject.assert_called_once()

    @patch("acestep.training.trainer.check_lycoris_available", return_value=True)
    @patch("acestep.training.trainer.inject_lokr_into_dit")
    def test_lokr_module_handles_inference_tensors(self, mock_inject, mock_lycoris):
        mock_inject.return_value = (self.model, MagicMock(), {"trainable_params": 100})

        lokr_config = LoKRConfig()
        training_config = TrainingConfig(output_dir="/tmp/out")

        # Create an inference mode tensor for one parameter
        with torch.inference_mode():
            inf_param = torch.randn(128, 128)
            self.model.decoder.q_proj.weight.data = inf_param

        self.assertTrue(self.model.decoder.q_proj.weight.is_inference())

        # This should not raise "inference mode tensors do not allow storage access"
        module = PreprocessedLoKRModule(
            model=self.model,
            lokr_config=lokr_config,
            training_config=training_config,
            device="cpu",
            dtype=torch.float32
        )

        self.assertFalse(self.model.decoder.q_proj.weight.is_inference())
        mock_inject.assert_called_once()

if __name__ == "__main__":
    unittest.main()
