
import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from acestep.training.trainer import PreprocessedLoRAModule, _LastLossAccessor, sample_continuous_timesteps
from acestep.training.configs import LoRAConfig, TrainingConfig

class TestTrainerLogic(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock(spec=nn.Module)
        self.model.config = MagicMock()
        self.model.decoder = MagicMock()
        self.model.null_condition_emb = torch.randn(1, 77, 1024)

        self.lora_config = LoRAConfig()
        self.training_config = TrainingConfig()
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def test_last_loss_accessor(self):
        container = MagicMock()
        accessor = _LastLossAccessor(container)

        self.assertFalse(accessor)
        self.assertEqual(len(accessor), 0)

        accessor.append(0.5)
        self.assertTrue(accessor)
        self.assertEqual(len(accessor), 1)
        self.assertEqual(accessor[-1], 0.5)
        self.assertEqual(container.last_training_loss, 0.5)

        accessor.append(0.7)
        self.assertEqual(accessor[-1], 0.7)
        self.assertEqual(container.last_training_loss, 0.7)

    def test_sample_continuous_timesteps(self):
        batch_size = 4
        t, r = sample_continuous_timesteps(
            batch_size, self.device, self.dtype,
            timestep_mu=-0.4, timestep_sigma=1.0, data_proportion=1.0
        )

        self.assertEqual(t.shape, (batch_size,))
        self.assertEqual(r.shape, (batch_size,))
        # With data_proportion=1.0, r should equal t
        torch.testing.assert_close(t, r)

        # With data_proportion=0.0, r should be <= t (due to max/min logic)
        t2, r2 = sample_continuous_timesteps(
            batch_size, self.device, self.dtype,
            timestep_mu=-0.4, timestep_sigma=1.0, data_proportion=0.0
        )
        self.assertTrue(torch.all(r2 <= t2))

    def test_lora_module_init(self):
        module = PreprocessedLoRAModule(
            self.model, self.lora_config, self.training_config, self.device, self.dtype
        )
        self.assertIsInstance(module.training_losses, _LastLossAccessor)
        self.assertEqual(module.last_training_loss, 0.0)
        self.assertIsNotNone(module._null_cond_emb)

if __name__ == "__main__":
    unittest.main()
