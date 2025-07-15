"""Simple generation logger to capture prompts and responses during PPO training."""

import json
from datetime import datetime
from pathlib import Path
import torch


class GenerationLogger:
    """Log prompts and responses during PPO training."""
    
    def __init__(self, log_dir: str = "logs", tokenizer=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        
        # Setup log files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.text_log = self.log_dir / f'generations_{timestamp}.log'
        self.json_log = self.log_dir / f'generations_{timestamp}.json'
        
        self.generations = []
        self.step = 0
        
        print(f"📝 Generation logger initialized: {self.text_log}")
    
    def log_step(self, queries, responses, rewards=None):
        """Log a training step with prompts and responses."""
        try:
            self.step += 1
            
            # Convert tensors to text
            prompts = []
            response_texts = []
            
            if self.tokenizer and hasattr(queries, '__iter__'):
                for query in queries:
                    if isinstance(query, torch.Tensor):
                        prompt = self.tokenizer.decode(query, skip_special_tokens=True)
                        prompts.append(prompt)
                    else:
                        prompts.append(str(query))
            
            if self.tokenizer and hasattr(responses, '__iter__'):
                for response in responses:
                    if isinstance(response, torch.Tensor):
                        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
                        response_texts.append(response_text)
                    else:
                        response_texts.append(str(response))
            
            # Extract rewards
            reward_values = []
            if rewards is not None:
                if isinstance(rewards, torch.Tensor):
                    reward_values = rewards.cpu().numpy().tolist()
                elif hasattr(rewards, '__iter__'):
                    reward_values = [float(r) for r in rewards]
                else:
                    reward_values = [float(rewards)]
            
            # Log to text file
            with open(self.text_log, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Step {self.step} - {datetime.now().strftime('%H:%M:%S')}\n")
                f.write(f"{'='*60}\n")
                
                for i, (prompt, response) in enumerate(zip(prompts, response_texts)):
                    reward = reward_values[i] if i < len(reward_values) else "N/A"
                    f.write(f"\nGeneration {i+1} | Reward: {reward}\n")
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Response: {response}\n")
                    f.write("-" * 40 + "\n")
            
            # Store for JSON
            step_data = {
                "step": self.step,
                "timestamp": datetime.now().isoformat(),
                "generations": []
            }
            
            for i, (prompt, response) in enumerate(zip(prompts, response_texts)):
                gen_data = {
                    "id": i,
                    "prompt": prompt,
                    "response": response
                }
                if i < len(reward_values):
                    gen_data["reward"] = reward_values[i]
                step_data["generations"].append(gen_data)
            
            self.generations.append(step_data)
            
            # Save JSON periodically
            if self.step % 5 == 0:
                self.save_json()
            
            # Print summary to console
            print(f"📝 Step {self.step}: Logged {len(prompts)} prompts and {len(response_texts)} responses")
            if reward_values:
                avg_reward = sum(reward_values) / len(reward_values)
                print(f"   Average reward: {avg_reward:.4f}")
        
        except Exception as e:
            print(f"⚠️  Generation logging error: {e}")
            # Log the error but don't break training
            with open(self.text_log, 'a') as f:
                f.write(f"\nERROR in step {self.step}: {e}\n")
    
    def save_json(self):
        """Save all generations to JSON file."""
        try:
            with open(self.json_log, 'w', encoding='utf-8') as f:
                json.dump(self.generations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  JSON save error: {e}")
    
    def close(self):
        """Finalize logging."""
        self.save_json()
        print(f"📝 Generation logging complete. {len(self.generations)} steps logged.")


def patch_ppo_trainer_for_generations(trainer, tokenizer, log_dir="logs"):
    """Patch PPO trainer to log generations."""
    
    gen_logger = GenerationLogger(log_dir, tokenizer)
    
    # Store original step method
    original_step = trainer.step
    
    def step_with_generation_logging(*args, **kwargs):
        # Call original step
        result = original_step(*args, **kwargs)
        
        # Log generations if we have the right arguments
        if len(args) >= 3:
            queries, responses, scores = args[0], args[1], args[2]
            gen_logger.log_step(queries, responses, scores)
        
        return result
    
    # Replace step method
    trainer.step = step_with_generation_logging
    trainer._generation_logger = gen_logger  # Store reference for cleanup
    
    return trainer
