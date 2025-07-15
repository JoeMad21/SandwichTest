import trl.trainer.utils as trl_utils
import trl.trainer.ppo_trainer as ppo_trainer_module
from train.patches.generation_patches import create_ultra_safe_batch_generation


def patch_trl_batch_generation():
    """Patch TRL's batch generation to use our safe version."""
    
    # Store original for potential use
    if not hasattr(trl_utils, '_original_batch_generation'):
        trl_utils._original_batch_generation = trl_utils.batch_generation
    
    # Create and apply the patch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")  # Temporary for the patch
    
    safe_batch_gen = create_ultra_safe_batch_generation(tokenizer)
    
    # Patch in multiple locations
    trl_utils.batch_generation = safe_batch_gen
    
    # Also patch in PPO trainer module if it exists
    if hasattr(ppo_trainer_module, 'batch_generation'):
        ppo_trainer_module.batch_generation = safe_batch_gen
    
    print("🔧 Patched TRL batch generation methods")
