"""Metrics tracking and aggregation for training."""

from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np


class MetricsTracker:
    """Track and aggregate training metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.step_metrics = defaultdict(list)
        
    def add_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Add a scalar metric value."""
        self.metrics[name].append(value)
        if step is not None:
            self.step_metrics[name].append((step, value))
    
    def add_batch_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """Add multiple metrics at once."""
        for name, value in metrics_dict.items():
            self.add_scalar(name, value, step)
    
    def get_average(self, name: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric over last_n values."""
        values = self.metrics.get(name, [])
        if not values:
            return 0.0
        
        if last_n is not None:
            values = values[-last_n:]
        
        return np.mean(values)
    
    def get_metrics_summary(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get summary of all metrics."""
        summary = {}
        for name in self.metrics:
            summary[f"{name}_avg"] = self.get_average(name, last_n)
            if self.metrics[name]:
                summary[f"{name}_last"] = self.metrics[name][-1]
        return summary
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.step_metrics.clear()


class RewardTracker:
    """Specialized tracker for reward metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards = []
        self.episode_rewards = []
        
    def add_reward(self, reward: float):
        """Add a single reward."""
        self.rewards.append(reward)
        
    def end_episode(self):
        """Mark end of episode and calculate episode metrics."""
        if self.rewards:
            episode_sum = sum(self.rewards)
            self.episode_rewards.append(episode_sum)
            self.rewards = []
            return episode_sum
        return 0.0
    
    def get_moving_average(self) -> float:
        """Get moving average of episode rewards."""
        if not self.episode_rewards:
            return 0.0
        
        window = self.episode_rewards[-self.window_size:]
        return np.mean(window)
    
    def get_stats(self) -> Dict[str, float]:
        """Get comprehensive reward statistics."""
        if not self.episode_rewards:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "moving_avg": 0.0
            }
        
        return {
            "mean": np.mean(self.episode_rewards),
            "std": np.std(self.episode_rewards),
            "min": np.min(self.episode_rewards),
            "max": np.max(self.episode_rewards),
            "moving_avg": self.get_moving_average()
        }
