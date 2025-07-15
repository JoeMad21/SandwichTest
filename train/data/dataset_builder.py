import math
from datasets import Dataset, Features, Sequence, Value
from train.generation_utils import clean_prompt


def create_train_dataset(num_episodes, tokenizer):
    """Create training dataset with prompts."""
    base_prompts = [
        "def sum_numbers(nums): # Example: sum_numbers([1,2,3]) -> 6",
        "def compute_sum(values): # Example: compute_sum([4,5]) -> 9",
        "def add_numbers(nums): # Example: add_numbers([0,1]) -> 1",
        "def calculate_total(data): # Example: calculate_total([10]) -> 10",
    ]
    
    # Repeat prompts to match num_episodes
    prompts = (base_prompts * math.ceil(num_episodes / len(base_prompts)))[:num_episodes]
    
    # Tokenize prompts
    queries = [
        tokenizer(clean_prompt(p), return_tensors="pt")["input_ids"][0]
        for p in prompts
    ]
    
    # Create dataset
    train_dataset = Dataset.from_dict(
        {"input_ids": [q.tolist() for q in queries]},
        features=Features({"input_ids": Sequence(Value("int64"))}),
    )
    
    return train_dataset


def create_eval_dataset(tokenizer):
    """Create evaluation dataset."""
    eval_prompts = ["def multiply_numbers(nums):", "def find_average(values):"]
    
    tokenized_eval = [
        tokenizer(clean_prompt(p), return_tensors="pt", padding=True)
        for p in eval_prompts
    ]
    
    eval_input_ids = [e["input_ids"][0].tolist() for e in tokenized_eval]
    eval_attention_mask = [e["attention_mask"][0].tolist() for e in tokenized_eval]
    
    eval_dataset = Dataset.from_dict(
        {
            "input_ids": eval_input_ids,
            "attention_mask": eval_attention_mask,
        },
        features=Features({
            "input_ids": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
        }),
    )
    
    return eval_dataset
