#!/usr/bin/env python3
"""
Real-time generation viewer for PPO training.
Usage: python scripts/watch_generations.py [--follow]
"""

import argparse
import time
import json
import os
from pathlib import Path


def get_latest_generation_file(log_dir: str = "logs"):
    """Get the most recent generation log file."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return None
    
    json_files = list(log_path.glob("generations_*.json"))
    if not json_files:
        return None
    
    return max(json_files, key=os.path.getctime)


def display_generations(data, n: int = 3):
    """Display the latest generations."""
    if not data:
        print("No generations recorded yet")
        return
    
    latest_batch = data[-1]
    step = latest_batch.get("step", "N/A")
    timestamp = latest_batch.get("timestamp", "")
    generations = latest_batch.get("generations", [])
    
    if not generations:
        print(f"Step {step}: No generations in latest batch")
        return
    
    # Clear screen for better viewing
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print(f"🤖 PPO Training - Generation Monitor")
    print(f"⏰ Time: {timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp}")
    print(f"📊 Step: {step} | Total Batches: {len(data)}")
    print("=" * 80)
    
    # Show latest n generations
    recent_gens = generations[-n:] if len(generations) > n else generations
    
    for i, gen in enumerate(recent_gens):
        reward = gen.get('reward', 'N/A')
        reward_str = f"Reward: {reward:.4f}" if isinstance(reward, (int, float)) else f"Reward: {reward}"
        
        print(f"📝 Generation {gen.get('id', i)} | {reward_str}")
        print(f"💭 Prompt: {gen['prompt']}")
        print(f"🤖 Response: {gen['response']}")
        print("-" * 40)
    
    # Show reward stats if available
    all_rewards = []
    for batch in data:
        for g in batch.get("generations", []):
            if "reward" in g and isinstance(g["reward"], (int, float)):
                all_rewards.append(g["reward"])
    
    if all_rewards:
        mean_reward = sum(all_rewards) / len(all_rewards)
        min_reward = min(all_rewards)
        max_reward = max(all_rewards)
        print(f"📈 Reward Stats: Mean={mean_reward:.4f}, Range=[{min_reward:.4f}, {max_reward:.4f}], Count={len(all_rewards)}")
    
    print("\n⌨️  Press Ctrl+C to exit")


def watch_generations(log_dir: str = "logs", follow: bool = False, n: int = 3):
    """Watch generations in real-time."""
    
    if follow:
        print("🔍 Watching for new generations... (Press Ctrl+C to stop)")
        last_modified = 0
        
        try:
            while True:
                gen_file = get_latest_generation_file(log_dir)
                
                if gen_file and gen_file.exists():
                    current_modified = gen_file.stat().st_mtime
                    
                    if current_modified > last_modified:
                        try:
                            with open(gen_file, 'r') as f:
                                data = json.load(f)
                            display_generations(data, n)
                            last_modified = current_modified
                        except (json.JSONDecodeError, FileNotFoundError):
                            # File might be being written to
                            pass
                
                time.sleep(2)  # Check every 2 seconds
                
        except KeyboardInterrupt:
            print("\n👋 Stopped watching generations")
    
    else:
        # Just show current state once
        gen_file = get_latest_generation_file(log_dir)
        
        if not gen_file:
            print(f"No generation files found in {log_dir}")
            print("Make sure training is running and generation monitoring is enabled.")
            return
        
        try:
            with open(gen_file, 'r') as f:
                data = json.load(f)
            display_generations(data, n)
        except Exception as e:
            print(f"Error reading generation file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Watch PPO training generations")
    parser.add_argument("--log-dir", "-d", default="logs", help="Log directory path")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow new generations")
    parser.add_argument("--count", "-n", type=int, default=3, help="Number of generations to show")
    
    args = parser.parse_args()
    
    watch_generations(args.log_dir, args.follow, args.count)


if __name__ == "__main__":
    main()
