"""Monitor and log model generations during training."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import torch


class GenerationMonitor:
    """Monitor and log model generations during PPO training."""
    
    def __init__(self, log_dir: str = "logs", save_frequency: int = 1):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        
        # Setup generation-specific logger
        self.logger = logging.getLogger("generation_monitor")
        if not self.logger.handlers:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            handler = logging.FileHandler(
                self.log_dir / f'generations_{timestamp}.log'
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Storage for generations
        self.generations_history = []
        self.current_step = 0
        
        # Setup JSON output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.json_file = self.log_dir / f'generations_{timestamp}.json'
        
    def log_generation_batch(self, 
                           step: int,
                           prompts: List[str], 
                           responses: List[str],
                           rewards: Optional[List[float]] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Log a batch of generations."""
        self.current_step = step
        
        # Create generation record
        generation_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "generations": []
        }
        
        if metadata:
            generation_data["metadata"] = metadata
        
        # Process each prompt-response pair
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            gen_record = {
                "id": i,
                "prompt": prompt.strip(),
                "response": response.strip(),
            }
            
            if rewards and i < len(rewards):
                gen_record["reward"] = rewards[i]
                
            generation_data["generations"].append(gen_record)
            
            # Log to text file for easy reading
            reward_str = f" | Reward: {rewards[i]:.4f}" if rewards and i < len(rewards) else ""
            self.logger.info(f"Step {step} | Gen {i}{reward_str}")
            self.logger.info(f"  Prompt: {prompt.strip()}")
            self.logger.info(f"  Response: {response.strip()}")
            self.logger.info("-" * 50)
        
        # Store in history
        self.generations_history.append(generation_data)
        
        # Save to JSON periodically
        if step % self.save_frequency == 0:
            self.save_generations_json()
    
    def log_single_generation(self, 
                            step: int,
                            prompt: str, 
                            response: str,
                            reward: Optional[float] = None,
                            metadata: Optional[Dict[str, Any]] = None):
        """Log a single generation."""
        self.log_generation_batch(
            step, [prompt], [response], 
            [reward] if reward is not None else None,
            metadata
        )
    
    def save_generations_json(self):
        """Save all generations to JSON file."""
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.generations_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save generations JSON: {e}")
    
    def get_latest_generations(self, n: int = 5) -> List[Dict]:
        """Get the most recent n generations."""
        if not self.generations_history:
            return []
        
        latest_batch = self.generations_history[-1]
        generations = latest_batch.get("generations", [])
        return generations[-n:] if len(generations) > n else generations
    
    def print_latest_generations(self, n: int = 3):
        """Print latest generations to console."""
        latest = self.get_latest_generations(n)
        if not latest:
            print("No generations recorded yet")
            return
        
        print(f"\n📝 Latest {len(latest)} Generation(s) (Step {self.current_step}):")
        print("=" * 60)
        
        for gen in latest:
            reward_str = f" | Reward: {gen.get('reward', 'N/A')}"
            print(f"Gen {gen['id']}{reward_str}")
            print(f"  📝 Prompt: {gen['prompt']}")
            print(f"  🤖 Response: {gen['response']}")
            print("-" * 40)
    
    def get_reward_stats(self) -> Dict[str, float]:
        """Get statistics about rewards across all generations."""
        all_rewards = []
        for batch in self.generations_history:
            for gen in batch.get("generations", []):
                if "reward" in gen:
                    all_rewards.append(gen["reward"])
        
        if not all_rewards:
            return {"count": 0}
        
        return {
            "count": len(all_rewards),
            "mean": sum(all_rewards) / len(all_rewards),
            "min": min(all_rewards),
            "max": max(all_rewards),
            "latest": all_rewards[-1] if all_rewards else None
        }


def patch_ppo_trainer_for_generation_logging(trainer, monitor: GenerationMonitor):
    """Patch PPO trainer to log generations."""
    original_step = trainer.step
    
    def step_with_logging(queries, responses, scores, response_masks=None):
        # Call original step
        result = original_step(queries, responses, scores, response_masks)
        
        # Extract data for logging
        try:
            # Convert tensors to strings
            prompts = []
            response_texts = []
            
            if hasattr(trainer, 'processing_class'):
                tokenizer = trainer.processing_class
                
                # Decode queries (prompts)
                if isinstance(queries, torch.Tensor):
                    for query in queries:
                        prompt = tokenizer.decode(query, skip_special_tokens=True)
                        prompts.append(prompt)
                
                # Decode responses
                if isinstance(responses, torch.Tensor):
                    for response in responses:
                        response_text = tokenizer.decode(response, skip_special_tokens=True)
                        response_texts.append(response_text)
            
            # Extract rewards/scores
            rewards = None
            if isinstance(scores, torch.Tensor):
                rewards = scores.cpu().numpy().tolist()
            elif isinstance(scores, (list, tuple)):
                rewards = [float(s) for s in scores]
            
            # Log the generation batch
            if prompts and response_texts:
                monitor.log_generation_batch(
                    step=trainer.state.episode if hasattr(trainer.state, 'episode') else 0,
                    prompts=prompts,
                    responses=response_texts,
                    rewards=rewards,
                    metadata={
                        "learning_rate": trainer.lr_scheduler.get_last_lr()[0] if trainer.lr_scheduler else None,
                        "global_step": trainer.state.global_step if hasattr(trainer.state, 'global_step') else None
                    }
                )
                
                # Print latest generations to console occasionally
                if trainer.state.episode % 5 == 0:  # Every 5 episodes
                    monitor.print_latest_generations(2)
        
        except Exception as e:
            monitor.logger.error(f"Failed to log generation: {e}")
        
        return result
    
    # Replace the step method
    trainer.step = step_with_logging
    return trainer


# Console generation viewer utility
def view_generations(log_dir: str = "logs", live: bool = False, n: int = 10):
    """View recent generations from log files."""
    log_path = Path(log_dir)
    
    # Find latest generation JSON file
    json_files = list(log_path.glob("generations_*.json"))
    if not json_files:
        print(f"No generation files found in {log_dir}")
        return
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            print("No generations recorded yet")
            return
        
        # Get latest batch
        latest_batch = data[-1]
        generations = latest_batch.get("generations", [])
        
        print(f"📝 Latest Generations (Step {latest_batch.get('step', 'N/A')}):")
        print("=" * 60)
        
        # Show last n generations
        recent_gens = generations[-n:] if len(generations) > n else generations
        
        for gen in recent_gens:
            reward_str = f" | Reward: {gen.get('reward', 'N/A')}"
            print(f"Gen {gen['id']}{reward_str}")
            print(f"  📝 Prompt: {gen['prompt']}")
            print(f"  🤖 Response: {gen['response']}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error reading generations: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="View model generations")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--live", action="store_true", help="Live view (not implemented)")
    parser.add_argument("-n", type=int, default=5, help="Number of generations to show")
    
    args = parser.parse_args()
    view_generations(args.log_dir, args.live, args.n)
