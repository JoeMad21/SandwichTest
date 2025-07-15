from torch.utils.data import DataLoader


def validate_eval_dataloader(eval_dataset, eval_collator):
    """Validate that eval dataloader works correctly."""
    sanity_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=eval_collator)
    
    print("🔍 Checking eval dataset batches...")
    for i, batch in enumerate(sanity_loader):
        assert batch is not None, f"Eval batch {i} is None"
        assert "input_ids" in batch, f"Eval batch {i} missing 'input_ids'"
        print(f"✅ Eval batch {i} passes: shape = {batch['input_ids'].shape}")
    
    print("✅ Eval dataloader structure confirmed.\n")


def create_eval_dataloader(eval_dataset, eval_collator, model):
    """Create eval dataloader with proper device handling."""
    target_device = next(model.parameters()).device
    
    def move_to_device_collate(batch):
        padded = eval_collator(batch)
        return {k: v.to(target_device) for k, v in padded.items()}
    
    return DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=move_to_device_collate,
    )
