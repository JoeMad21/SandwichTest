import torch
import gc

def clean_prompt(prompt: str) -> str:
    for phrase in ["Complete the Python function:\n", "Complete Python function body:\n"]:
        prompt = prompt.replace(phrase, "")
    return prompt.strip()

def clean_code(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        text = text[len(prompt):]
    lines = text.strip().split("\n")
    return "\n".join(
        line for line in lines
        if line.strip() and not line.strip().startswith("#")
    )

def safe_generate(model, tokenizer, prompt: str, max_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gc.collect()

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        return output[0], decoded
