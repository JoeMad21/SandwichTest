import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig

def load_model_and_tokenizer(model_name: str, **model_kwargs):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # BitsAndBytes config for 4-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=bnb_config,
        device_map={"":0}
    )

    # FIXED: Pass tokenizer vocab size to value head function
    from train.value_head_patch import add_value_head
    tokenizer_vocab_size = len(tokenizer)
    model = add_value_head(model, tokenizer_vocab_size)

    model.return_dict = True
    if not hasattr(model, "generation_config"):
        model.generation_config = GenerationConfig.from_pretrained(model_name)
    
    # Set pad token if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Ensure model config matches tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer
