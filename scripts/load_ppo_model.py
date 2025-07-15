#!/usr/bin/env python3
"""
Load the PPO model correctly by handling the value head.
"""

import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def inspect_model_files(model_path="./ppo_model_final"):
    """Inspect the saved model files."""
    model_path = Path(model_path)
    
    print("📁 Model files:")
    for file in sorted(model_path.glob("*")):
        size = file.stat().st_size
        print(f"   {file.name}: {size:,} bytes")
    
    # Check config
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"\n📄 Model architecture: {config.get('architectures', 'Unknown')}")
        print(f"📊 Vocab size in config: {config.get('vocab_size', 'Unknown')}")
        print(f"🏷️  Pad token ID in config: {config.get('pad_token_id', 'Unknown')}")
    
    # Check generation config
    gen_config_file = model_path / "generation_config.json"
    if gen_config_file.exists():
        with open(gen_config_file) as f:
            gen_config = json.load(f)
        print(f"\n🎛️  Generation config:")
        for key, value in gen_config.items():
            print(f"   {key}: {value}")


def load_ppo_model_correctly(model_path="./ppo_model_final"):
    """Load the PPO model by extracting just the language model part."""
    
    print(f"🔄 Loading PPO model from {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded. Vocab size: {len(tokenizer)}")
        
        # Method 1: Try loading with ignore_mismatched_sizes
        print("\n🔄 Attempting to load with ignore_mismatched_sizes=True...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                ignore_mismatched_sizes=True,  # Ignore the value head
                trust_remote_code=True
            )
            print("✅ Model loaded successfully (ignoring mismatched sizes)")
            
            # Set pad token correctly
            if tokenizer.pad_token_id != model.config.pad_token_id:
                print(f"🔧 Fixing pad token mismatch: {tokenizer.pad_token_id} -> {model.config.pad_token_id}")
                model.config.pad_token_id = tokenizer.pad_token_id
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Method 1 failed: {e}")
        
        # Method 2: Try loading the base model and copying weights manually
        print("\n🔄 Attempting to load base model and copy weights...")
        try:
            # Load base model architecture
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            # Load the state dict from our trained model
            state_dict = torch.load(model_path / "model.safetensors", map_location="cpu")
            
            # Filter out value head weights
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('v_head'):  # Skip value head weights
                    # Remove any prefix if present
                    clean_key = key.replace('model.', '') if key.startswith('model.') else key
                    filtered_state_dict[clean_key] = value
            
            # Load the filtered weights
            missing_keys, unexpected_keys = base_model.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
            
            print("✅ Weights copied to base model architecture")
            
            # Update config
            base_model.config.pad_token_id = tokenizer.pad_token_id
            base_model.config.vocab_size = len(tokenizer)
            
            return base_model, tokenizer
            
        except Exception as e:
            print(f"❌ Method 2 failed: {e}")
            
        return None, None
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None


def test_model_generation(model, tokenizer):
    """Test the loaded model."""
    
    if model is None or tokenizer is None:
        print("❌ No model to test")
        return
    
    print("\n🧪 Testing model generation...")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    test_prompts = [
        "def sum_numbers(nums):",
        "def add_numbers(nums):",
        "def hello():",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Testing: {prompt}")
        
        try:
            # Encode
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with very conservative settings
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,  # Greedy only
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False  # Disable cache
                )
            
            # Decode
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_part = generated[len(prompt):].strip()
            
            print(f"   ✅ Generated: {prompt}{new_part}")
            
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")


def main():
    print("🔧 PPO Model Loader")
    print("=" * 50)
    
    model_path = "./ppo_model_final"
    
    # First inspect the files
    inspect_model_files(model_path)
    
    print("\n" + "=" * 50)
    
    # Try to load the model
    model, tokenizer = load_ppo_model_correctly(model_path)
    
    if model is not None:
        # Test generation
        test_model_generation(model, tokenizer)
    else:
        print("\n❌ Could not load the model. The PPO training may have corrupted it.")
        print("\n💡 Suggestions:")
        print("1. The model architecture changed during PPO training")
        print("2. Try using the base microsoft/phi-2 model instead")
        print("3. Check if training completed successfully")


if __name__ == "__main__":
    main()
