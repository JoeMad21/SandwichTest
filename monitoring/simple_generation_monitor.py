"""Simple generation monitor that hooks into TRL callbacks."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import torch


class SimpleGenerationMonitor:
    """Simple monitor for PPO generations using TRL callbacks."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup generation logger
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.gen_log_file = self.log_dir / f'generations_{timestamp}.log'
        self.json_file = self.log_dir / f'generations_{timestamp}.json'
        
        # Simple file logging
        self.generations = []
        self.step_count = 0
        
        print(f"📝 Generation monitor initialized: {self.gen_log_file}")
    
    def log_batch(self, step: int, queries, responses, scores=None):
        """Log a batch of generations with minimal processing."""
        try:
            self.step_count = step
            
            # Simple logging to file
            with open(self.gen_log_file, 'a') as f:
                f.write(f"\n=== Step {step} ===\n")
                
                # Try to decode if possible
                if hasattr(queries, '__len__') and hasattr(responses, '__len__'):
                    for i in range(min(len(queries), len(responses))):
                        query_text = str(queries[i]) if queries[i] is not None else "N/A"
                        response_text = str(responses[i]) if responses[i] is not None else "N/A"
                        score = scores[i] if scores and i < len(scores) else "N/A"
                        
                        f.write(f"Gen {i} | Score: {score}\n")
                        f.write(f"  Query: {query_text[:100]}...\n")
                        f.write(f"  Response: {response_text[:100]}...\n")
                        f.write("-" * 40 + "\n")
                
                f.flush()
            
            # Store for JSON
            batch_data = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "count": len(queries) if hasattr(queries, '__len__') else 1
            }
            self.generations.append(batch_data)
            
            # Save JSON periodically
            if step % 5 == 0:
                self.save_json()
                
            # Print to console occasionally
            if step % 10 == 0:
                print(f"📝 Step {step}: Logged {batch_data['count']} generations")
                
        except Exception as e:
            print(f"⚠️  Generation logging error: {e}")
    
    def save_json(self):
        """Save generation metadata to JSON."""
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.generations, f, indent=2)
        except Exception as e:
            print(f"⚠️  JSON save error: {e}")


class PPOCallback:
    """Simple callback for PPO trainer."""
    
    def __init__(self, monitor: SimpleGenerationMonitor):
        self.monitor = monitor
        self.step = 0
    
    def on_step_end(self, trainer, queries, responses, scores, **kwargs):
        """Called after each PPO step."""
        self.step += 1
        self.monitor.log_batch(self.step, queries, responses, scores)


def add_simple_monitoring(trainer, log_dir: str = "logs"):
    """Add simple monitoring to PPO trainer without complex patching."""
    monitor = SimpleGenerationMonitor(log_dir)
    
    # Store original step method
    original_step = trainer.step
    
    def step_with_logging(*args, **kwargs):
        # Call original step
        result = original_step(*args, **kwargs)
        
        # Simple logging
        if len(args) >= 3:  # queries, responses, scores
            queries, responses, scores = args[0], args[1], args[2]
            episode = getattr(trainer.state, 'episode', 0) if hasattr(trainer, 'state') else 0
            monitor.log_batch(episode, queries, responses, scores)
        
        return result
    
    # Replace step method
    trainer.step = step_with_logging
    
    return monitor
