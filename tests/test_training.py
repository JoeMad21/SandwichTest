"""Tests for training components."""

import pytest
from pathlib import Path
import tempfile
import torch

from configs.training_config import TrainingConfig
from configs.model_config import ModelConfig
from train.data.dataset_builder import create_train_dataset, create_eval_dataset
from train.safety.numerical_stability import check_model_stability
from utils.validation import ConfigValidator
from monitoring.metrics import MetricsTracker, RewardTracker


class TestConfigs:
    """Test configuration classes."""
    
    def test_training_config_from_yaml(self, tmp_path):
        """Test loading training config from YAML."""
        # Create temporary YAML file
        yaml_content = """
learning_rate: 5e-7
batch_size: 1
mini_batch_size: 1
gradient_accumulation_steps: 2
total_episodes: 10
logging_steps: 5
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        
        # Load config
        config = TrainingConfig.from_yaml(str(yaml_file))
        assert config.learning_rate == 5e-7
        assert config.batch_size == 1
        assert config.total_episodes == 10
    
    def test_model_config_defaults(self):
        """Test model config default values."""
        config = ModelConfig()
        assert config.model_name == "microsoft/phi-2"
        assert config.use_4bit is True
        assert config.device_map == {"": 0}
    
    def test_config_validation(self):
        """Test configuration validation."""
        valid_config = {
            'learning_rate': 1e-6,
            'batch_size': 2,
            'total_episodes': 100,
        }
        
        validated = ConfigValidator.validate_ppo_config(valid_config)
        assert validated['learning_rate'] == 1e-6
        assert validated['batch_size'] == 2
        
        # Test invalid config
        invalid_config = {
            'learning_rate': 2,  # Too high
            'batch_size': -1,    # Negative
            'total_episodes': 100,
        }
        
        with pytest.raises(ValueError):
            ConfigValidator.validate_ppo_config(invalid_config)


class TestDatasetBuilding:
    """Test dataset creation."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                # Simple mock: return tensor of word count
                return {"input_ids": torch.tensor([list(range(len(text.split())))])}
        
        return MockTokenizer()
    
    def test_create_train_dataset(self, mock_tokenizer):
        """Test training dataset creation."""
        dataset = create_train_dataset(num_episodes=8, tokenizer=mock_tokenizer)
        
        assert len(dataset) == 8
        assert "input_ids" in dataset.features
        
        # Check that prompts are recycled correctly
        first_item = dataset[0]["input_ids"]
        fifth_item = dataset[4]["input_ids"]  # Should be same prompt
        assert first_item == fifth_item
    
    def test_create_eval_dataset(self, mock_tokenizer):
        """Test evaluation dataset creation."""
        dataset = create_eval_dataset(tokenizer=mock_tokenizer)
        
        assert len(dataset) == 2  # Two eval prompts
        assert "input_ids" in dataset.features
        assert "attention_mask" in dataset.features


class TestNumericalStability:
    """Test numerical stability checks."""
    
    def test_check_model_stability(self):
        """Test model stability checking."""
        # Create simple model
        model = torch.nn.Linear(10, 10)
        
        # Normal model should pass
        assert check_model_stability(model) is True
        
        # Add NaN to weights
        model.weight.data[0, 0] = float('nan')
        assert check_model_stability(model) is False
        
        # After check, NaN should be replaced
        assert not torch.isnan(model.weight.data[0, 0])


class TestMetrics:
    """Test metrics tracking."""
    
    def test_metrics_tracker(self):
        """Test basic metrics tracking."""
        tracker = MetricsTracker()
        
        # Add some metrics
        tracker.add_scalar("loss", 0.5, step=1)
        tracker.add_scalar("loss", 0.4, step=2)
        tracker.add_scalar("loss", 0.3, step=3)
        
        # Test average
        assert tracker.get_average("loss") == pytest.approx(0.4)
        assert tracker.get_average("loss", last_n=2) == pytest.approx(0.35)
        
        # Test summary
        summary = tracker.get_metrics_summary()
        assert "loss_avg" in summary
        assert "loss_last" in summary
        assert summary["loss_last"] == 0.3
    
    def test_reward_tracker(self):
        """Test reward tracking."""
        tracker = RewardTracker(window_size=3)
        
        # Add rewards for multiple episodes
        tracker.add_reward(1.0)
        tracker.add_reward(2.0)
        episode1_reward = tracker.end_episode()
        assert episode1_reward == 3.0
        
        tracker.add_reward(2.5)
        tracker.add_reward(2.5)
        tracker.end_episode()
        
        # Test stats
        stats = tracker.get_stats()
        assert stats["mean"] == pytest.approx(4.0)  # (3.0 + 5.0) / 2
        assert stats["min"] == 3.0
        assert stats["max"] == 5.0


@pytest.fixture
def cleanup_cuda():
    """Cleanup CUDA memory after tests."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
