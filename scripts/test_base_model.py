#!/usr/bin/env python3
"""
Test the base model to compare with PPO-trained model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_base_model():
    """Test the original microsoft/phi-2 model."""
    print("🔄 Loading base microsoft/phi-2 model...")
    
    try:
        # Load base model
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Base model loaded successfully")
        
        # Test prompts (same as your training data)
        test_prompts = [
            "def sum_numbers(nums): # Example: sum_numbers([1,2,3]) -> 6",
            "def hello():",
            "Hello, how are you?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}: {prompt}")
            
            try:
                # Encode
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate with conservative settings
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,  # Greedy decoding
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # Decode
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                new_part = generated[len(prompt):].strip()
                
                print(f"✅ Generated: {new_part}")
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Base model test failed: {e}")
        return False

if __name__ == "__main__":
    test_base_model()
