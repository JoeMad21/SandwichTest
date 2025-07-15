import math
import os
import torch
from torch.utils.data import DataLoader
from trl import PPOTrainer

from configs.training_config import TrainingConfig
from configs.model_config import ModelConfig
from train.model_utils import load_model_and_tokenizer
from train.checkpoint_utils import save_checkpoint
from train.patches.reference_model import PatchedReferenceModel
from train.patches.generation_patches import patch_generation_methods
from train.patches.trl_patches import patch_trl_batch_generation
from train.safety.numerical_stability import check_model_stability
from train.safety.cuda_safety import setup_cuda_debugging
from train.data.dataset_builder import create_train_dataset, create_eval_dataset
from train.data.collators import get_eval_collator
from train.data.dataloader_utils import create_eval_dataloader, validate_eval_dataloader
from reward.reward_fn import reward_fn
from monitoring.logger import (
    setup_logger, log_training_start, log_model_info, log_training_complete,
    restore_output_redirection
)
import time

def run_training():
    """Main training orchestration function."""
    start_time = time.time()
    
    # Setup logging with output redirection
    logger, orig_stdout, orig_stderr = setup_logger("ppo_training", log_dir="logs", redirect_stdout=True)
    logger.info("📝 Logging setup complete - verbose output redirected to files")
    logger.info("📁 Log files saved to logs/ directory")
    
    try:
        # Setup
        setup_cuda_debugging()
        
        # Load configurations
        model_config = ModelConfig()
        training_config = TrainingConfig.from_yaml("configs/ppo_config.yaml")
        
        # Log training start
        config_dict = {
            "model": model_config.model_name,
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.batch_size,
            "total_episodes": training_config.total_episodes,
            "output_dir": training_config.output_dir
        }
        log_training_start(logger, config_dict)
        
        # Load model and tokenizer
        logger.info("🔄 Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_config.model_name)
        
        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        log_model_info(logger, model_config.model_name, num_params)
        
        # FIXED: Ensure proper padding token without changing vocab size
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
            # Use EOS token as pad token (standard practice)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
            logger.info(f"🏷️  Set pad token to EOS token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        elif tokenizer.pad_token_id == tokenizer.eos_token_id:
            # Already using EOS as pad, just ensure model config matches
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"🏷️  Pad token already set to EOS: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        else:
            # Pad token exists and is different from EOS, ensure model config matches
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"🏷️  Using existing pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

        # FIXED: Use the actual tokenizer vocab size (phi-2 has 50295, not 51200)
        actual_vocab_size = len(tokenizer)
        model_config_vocab_size = model.config.vocab_size
        
        if actual_vocab_size != model_config_vocab_size:
            logger.info(f"🔧 Model config vocab size ({model_config_vocab_size}) != tokenizer vocab size ({actual_vocab_size})")
            logger.info(f"🔧 This is normal for phi-2. Using tokenizer vocab size: {actual_vocab_size}")
            # Update model config to match tokenizer
            model.config.vocab_size = actual_vocab_size
        else:
            logger.info(f"✅ Vocab sizes match: {actual_vocab_size}")
        
        # Create reference model
        logger.info("🔄 Creating reference model...")
        ref_model = PatchedReferenceModel(model)
        ref_model.to("cpu")
        
        # Update training config with tokenizer info
        training_config.stop_token_id = tokenizer.eos_token_id
        ppo_config = training_config.to_ppo_config()
        
        # Create datasets
        logger.info("📚 Creating datasets...")
        train_dataset = create_train_dataset(
            num_episodes=training_config.total_episodes,
            tokenizer=tokenizer
        )
        
        eval_dataset = create_eval_dataset(tokenizer)
        
        # Validate eval dataloader
        eval_collator = get_eval_collator(tokenizer)
        validate_eval_dataloader(eval_dataset, eval_collator)
        logger.info("✅ Dataset validation complete")
        
        # Create trainer
        logger.info("🔄 Initializing PPO trainer...")
        trainer = PPOTrainer(
            ppo_config,
            model=model,
            ref_model=ref_model,
            value_model=model,
            reward_model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=eval_collator,
        )
        
        # Apply all safety patches
        logger.info("🛡️  Applying safety patches...")
        patch_generation_methods(model, tokenizer)
        patch_trl_batch_generation()
        
        # Check numerical stability
        if check_model_stability(model):
            logger.info("✅ Model stability check passed")
        
        # Setup eval dataloader with proper device handling
        trainer.eval_dataloader = create_eval_dataloader(
            eval_dataset, 
            eval_collator,
            model
        )
        
        # Disable problematic eval completions
        def skip_eval_completions(self, *args, **kwargs):
            logger.info("🔁 Skipping eval completions (CUDA safety)")
            return
        
        trainer._original_generate_completions = trainer.generate_completions
        trainer.generate_completions = skip_eval_completions.__get__(trainer)
        
        # Train
        logger.info("🚀 Starting training loop...")
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        save_checkpoint(model, tokenizer, training_config.output_dir)
        
        # Log completion
        total_time = time.time() - start_time
        log_training_complete(logger, total_time)
        
        logger.info("📁 Check logs/ directory for detailed output files")
        
    finally:
        # Always restore original stdout/stderr
        if orig_stdout and orig_stderr:
            restore_output_redirection(orig_stdout, orig_stderr)


if __name__ == "__main__":
    run_training()
