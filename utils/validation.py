"""Input validation utilities."""

from typing import Any, List, Dict, Optional
from pathlib import Path


def validate_model_name(model_name: str) -> bool:
    """Validate model name format."""
    if not model_name or not isinstance(model_name, str):
        return False
    
    # Basic check for common model name formats
    parts = model_name.split("/")
    if len(parts) not in [1, 2]:
        return False
    
    return all(part.replace("-", "").replace("_", "").isalnum() for part in parts)


def validate_config_dict(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate configuration dictionary has required keys."""
    return all(key in config for key in required_keys)


def validate_file_path(path: Union[str, Path], must_exist: bool = False) -> bool:
    """Validate file path."""
    try:
        path = Path(path)
        if must_exist:
            return path.exists()
        return True
    except Exception:
        return False


def validate_positive_int(value: Any, name: str = "value") -> int:
    """Validate and convert to positive integer."""
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValueError(f"{name} must be positive")
        return int_value
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid {name}: {value}") from e


def validate_float_range(value: Any, min_val: float = 0.0, max_val: float = 1.0, name: str = "value") -> float:
    """Validate float is within range."""
    try:
        float_value = float(value)
        if not min_val <= float_value <= max_val:
            raise ValueError(f"{name} must be between {min_val} and {max_val}")
        return float_value
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid {name}: {value}") from e


def validate_tokenizer(tokenizer) -> bool:
    """Validate tokenizer has required attributes."""
    required_attrs = ['pad_token_id', 'eos_token_id', 'encode', 'decode']
    return all(hasattr(tokenizer, attr) for attr in required_attrs)


def validate_model(model) -> bool:
    """Validate model has required methods."""
    required_methods = ['forward', 'generate', 'parameters']
    return all(hasattr(model, method) for method in required_methods)


class ConfigValidator:
    """Validate training configuration."""
    
    @staticmethod
    def validate_ppo_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean PPO configuration."""
        validated = {}
        
        # Required fields
        validated['learning_rate'] = validate_float_range(
            config['learning_rate'], 1e-8, 1e-2, "learning_rate"
        )
        validated['batch_size'] = validate_positive_int(
            config['batch_size'], "batch_size"
        )
        validated['total_episodes'] = validate_positive_int(
            config['total_episodes'], "total_episodes"
        )
        
        # Optional fields with defaults
        validated['mini_batch_size'] = validate_positive_int(
            config.get('mini_batch_size', validated['batch_size']), 
            "mini_batch_size"
        )
        validated['gradient_accumulation_steps'] = validate_positive_int(
            config.get('gradient_accumulation_steps', 1),
            "gradient_accumulation_steps"
        )
        validated['logging_steps'] = validate_positive_int(
            config.get('logging_steps', 10),
            "logging_steps"
        )
        
        # Ensure mini_batch_size <= batch_size
        if validated['mini_batch_size'] > validated['batch_size']:
            validated['mini_batch_size'] = validated['batch_size']
        
        return validated
