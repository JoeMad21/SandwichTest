from typing import Dict, Any


def get_safe_generation_config(tokenizer) -> Dict[str, Any]:
    """Return ultra-safe generation configuration."""
    return {
        'max_new_tokens': 10,
        'do_sample': False,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'use_cache': True,
        'output_scores': False,
        'num_beams': 1,
    }


def validate_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure generation config is safe."""
    safe_config = config.copy()
    
    # Force safe values
    safe_config['do_sample'] = False
    safe_config['num_beams'] = 1
    safe_config['max_new_tokens'] = min(safe_config.get('max_new_tokens', 10), 10)
    
    # Remove dangerous parameters
    for key in ['temperature', 'top_k', 'top_p', 'typical_p']:
        safe_config.pop(key, None)
    
    return safe_config
