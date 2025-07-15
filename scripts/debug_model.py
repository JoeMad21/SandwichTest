#!/usr/bin/env python3
"""
Debug the trained model to identify issues.
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def check_model_health(model_path="./ppo_model_final"):
    """Check if the model has issues."""
    
    print("🔍 Debugging model health...")
    
    try:
        # Load tokenizer first
        print("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"   ✅ Tokenizer loaded. Vocab size: {len(tokenizer)}")
        print(f"   Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        
        # Try to load without quantization first
        print("\n2. Loading model without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Full precision
            device_map="cpu",  # CPU first to avoid CUDA issues
            quantization_config=None  # No quantization
        )
        print("   ✅ Model loaded on CPU")
        
        # Check model architecture
        print(f"   Model type: {type(model)}")
        print(f"   Config: {model.config.architectures}")
        
        # Check vocab size mismatch
        model_vocab_size = model.config.vocab_size
        tokenizer_vocab_size = len(tokenizer)
        print(f"   Model vocab size: {model_vocab_size}")
        print(f"   Tokenizer vocab size: {tokenizer_vocab_size}")
        
        if model_vocab_size != tokenizer_vocab_size:
            print("   ⚠️  VOCAB SIZE MISMATCH - This could cause CUDA errors!")
            print(f"   Difference: {abs(model_vocab_size - tokenizer_vocab_size)}")
        
        # Try a simple forward pass on CPU
        print("\n3. Testing simple forward pass on CPU...")
        test_input = "def test():"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            print(f"   ✅ Forward pass successful. Logits shape: {logits.shape}")
            
            # Check for NaN or inf in logits
            if torch.isnan(logits).any():
                print("   ❌ NaN values found in logits!")
            elif torch.isinf(logits).any():
                print("   ❌ Inf values found in logits!")
            else:
                print("   ✅ Logits are clean (no NaN/inf)")
        
        # Try moving to GPU
        print("\n4. Testing on GPU...")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                print("   ✅ GPU forward pass successful")
        except Exception as e:
            print(f"   ❌ GPU forward pass failed: {e}")
            return False
        
        # Try simple generation
        print("\n5. Testing simple generation...")
        try:
            with torch.no_grad():
                generated = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
                print(f"   ✅ Generation successful: '{decoded}'")
                return True
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def compare_with_base_model():
    """Compare with the original phi-2 model."""
    print("\n🆚 Comparing with base model...")
    
    try:
        print("Loading base microsoft/phi-2...")
        base_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        
        print("✅ Base model loaded")
        print(f"Base vocab size: {len(base_tokenizer)}")
        print(f"Base model vocab size: {base_model.config.vocab_size}")
        
        # Test generation with base model
        test_input = "def test():"
        inputs = base_tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            generated = base_model.generate(
                inputs['input_ids'],
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=base_tokenizer.pad_token_id
            )
            decoded = base_tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"Base model generation: '{decoded}'")
            
    except Exception as e:
        print(f"❌ Base model test failed: {e}")


def main():
    print("🔧 Model Debugging Tool")
    print("=" * 50)
    
    # Check if trained model exists
    model_path = "./ppo_model_final"
    if not Path(model_path).exists():
        print(f"❌ Model path {model_path} does not exist")
        return
    
    # Debug the trained model
    success = check_model_health(model_path)
    
    if not success:
        print("\n💡 Suggestions:")
        print("1. The model may have corrupted weights from PPO training")
        print("2. Vocab size mismatch between model and tokenizer")
        print("3. Try loading the base model instead")
        
        # Test base model
        compare_with_base_model()
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
