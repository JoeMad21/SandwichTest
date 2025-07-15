from transformers import DataCollatorWithPadding


def get_eval_collator(tokenizer):
    """Get data collator for evaluation."""
    return DataCollatorWithPadding(tokenizer)


def get_train_collator(tokenizer):
    """Get data collator for training."""
    return DataCollatorWithPadding(tokenizer, padding=True)
