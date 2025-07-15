#!/usr/bin/env python3
"""
Test the trained PPO model for inference.
Usage: python scripts/test_trained_model.py [--model-path ./ppo_model_final] [--interactive]
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


def load_trained_model(model_path: str):
    """Load the trained model and tokenizer with safer settings."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    
    print(f"🔄 Loading trained model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with safer settings
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True  # In case model has custom code
    )
    
    # FIXED: Don't override tokenizer settings - use what was saved during training
    # Just verify they exist and are consistent
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("⚠️  Warning: No pad token found in saved tokenizer")
        # Only set if truly missing
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Ensure model config matches tokenizer
    if model.config.pad_token_id != tokenizer.pad_token_id:
        print(f"🔧 Syncing model pad_token_id: {model.config.pad_token_id} -> {tokenizer.pad_token_id}")
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Set model to eval mode
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🏷️  Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"🏷️  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"📚 Vocab size: {len(tokenizer)}")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_length: int = 128, temperature: float = 0.7):
    """Generate a response to a prompt with safer generation settings."""
    
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    # Use safer generation settings to avoid CUDA errors
    with torch.no_grad():
        try:
            outputs = model.generate(
                inputs,
                max_new_tokens=64,  # Use max_new_tokens instead of max_length
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1,
                attention_mask=torch.ones_like(inputs),  # Explicit attention mask
                use_cache=True
            )
        except Exception as e:
            print(f"Generation failed, trying greedy decoding: {e}")
            # Fallback to greedy decoding if sampling fails
            outputs = model.generate(
                inputs,
                max_new_tokens=64,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs),
                use_cache=True
            )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part (after the prompt)
    generated_text = response[len(prompt):].strip()
    
    return generated_text


def test_sample_prompts(model, tokenizer):
    """Test the model with coding prompts it was trained on."""
    
    # Prompts based on your training data
    coding_prompts = [
        "def sum_numbers(nums): # Example: sum_numbers([1,2,3]) -> 6",
        "def compute_sum(values): # Example: compute_sum([4,5]) -> 9", 
        "def add_numbers(nums): # Example: add_numbers([0,1]) -> 1",
        "def calculate_total(data): # Example: calculate_total([10]) -> 10",
        "def multiply_numbers(nums):",
        "def find_average(values):",
        "def recursive_sum(lst):",
        "def sum_even_numbers(numbers):",
        "def sum_positive(values):",
        "def accumulate_values(data):",
        "def add(a, b):",
        "def process_list(items):",
        'def calculate(x):\n    """Calculate result"""',
        "def compute(values: list) -> int:",
    ]
    
    print("\n🤖 Testing trained model with coding prompts:")
    print("=" * 80)
    
    for i, prompt in enumerate(coding_prompts, 1):
        print(f"\n📝 Prompt {i}: {prompt}")
        print("-" * 60)
        
        try:
            response = generate_response(model, tokenizer, prompt, temperature=0.3)  # Lower temp for code
            print(f"🤖 Generated code:\n{prompt}{response}")
        except Exception as e:
            print(f"❌ Error generating response: {e}")
        
        print("-" * 60)


def interactive_mode(model, tokenizer):
    """Interactive chat mode with the trained model."""
    
    print("\n💬 Interactive mode - Chat with your trained model!")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\n📝 You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("🤖 Model: ", end="", flush=True)
            response = generate_response(model, tokenizer, prompt)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def compare_with_base_model(trained_model, trained_tokenizer, base_model_name: str, prompt: str):
    """Compare trained model response with base model."""
    
    print(f"\n🔄 Loading base model {base_model_name} for comparison...")
    
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        
        print(f"\n🆚 Comparing responses to: '{prompt}'")
        print("=" * 60)
        
        # Generate with base model
        print("📝 Base model response:")
        base_response = generate_response(base_model, base_tokenizer, prompt)
        print(f"   {base_response}")
        
        # Generate with trained model
        print("\n📝 Trained model response:")
        trained_response = generate_response(trained_model, trained_tokenizer, prompt)
        print(f"   {trained_response}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error loading base model: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test trained PPO model")
    parser.add_argument("--model-path", "-m", default="./ppo_model_final", 
                       help="Path to trained model directory")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive chat mode")
    parser.add_argument("--compare", "-c", 
                       help="Compare with base model (e.g., microsoft/phi-2)")
    parser.add_argument("--prompt", "-p",
                       help="Single prompt to test")
    parser.add_argument("--max-length", type=int, default=128,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    
    args = parser.parse_args()
    
    try:
        # Load the trained model
        model, tokenizer = load_trained_model(args.model_path)
        
        if args.prompt:
            # Test single prompt
            print(f"\n📝 Prompt: {args.prompt}")
            response = generate_response(model, tokenizer, args.prompt, 
                                       args.max_length, args.temperature)
            print(f"🤖 Response: {response}")
            
            # Compare with base model if requested
            if args.compare:
                compare_with_base_model(model, tokenizer, args.compare, args.prompt)
                
        elif args.interactive:
            # Interactive mode
            interactive_mode(model, tokenizer)
        else:
            # Test with sample prompts
            test_sample_prompts(model, tokenizer)
            
            # Compare with base model if requested
            if args.compare:
                compare_with_base_model(model, tokenizer, args.compare, 
                                      "def sum_numbers(nums):")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
