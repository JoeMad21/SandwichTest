import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # PPO specific
    learning_rate: float
    batch_size: int
    mini_batch_size: int
    gradient_accumulation_steps: int
    total_episodes: int
    logging_steps: int
    stop_token_id: int
    
    # Additional training params
    output_dir: str = "./ppo_model_final"
    max_new_tokens: int = 64
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Ensure learning_rate is a float
        if 'learning_rate' in config_dict:
            config_dict['learning_rate'] = float(config_dict['learning_rate'])
        
        return cls(**config_dict)
    
    def to_ppo_config(self):
        """Convert to PPO trainer configuration format"""
        from trl import PPOConfig
        
        return PPOConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            mini_batch_size=self.mini_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            total_episodes=self.total_episodes,
            logging_steps=self.logging_steps,
            stop_token_id=self.stop_token_id,
        )
