import os

def save_checkpoint(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_checkpoint(output_dir, model_cls, tokenizer_cls):
    model = model_cls.from_pretrained(output_dir)
    tokenizer = tokenizer_cls.from_pretrained(output_dir)
    return model, tokenizer
