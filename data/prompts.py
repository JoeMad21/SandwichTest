"""Training and evaluation prompts for the model."""

# Base training prompts focused on summation tasks
BASE_TRAINING_PROMPTS = [
    "def sum_numbers(nums): # Example: sum_numbers([1,2,3]) -> 6",
    "def compute_sum(values): # Example: compute_sum([4,5]) -> 9", 
    "def add_numbers(nums): # Example: add_numbers([0,1]) -> 1",
    "def calculate_total(data): # Example: calculate_total([10]) -> 10",
]

# Evaluation prompts to test generalization
EVAL_PROMPTS = [
    "def multiply_numbers(nums):",
    "def find_average(values):",
]

# Additional prompts for future expansion
ADVANCED_PROMPTS = [
    "def recursive_sum(lst):",
    "def sum_even_numbers(numbers):",
    "def sum_positive(values):",
    "def accumulate_values(data):",
]

# Prompts for testing specific behaviors
TEST_PROMPTS = {
    "simple": "def add(a, b):",
    "list_operation": "def process_list(items):",
    "with_docstring": 'def calculate(x):\n    """Calculate result"""',
    "with_type_hints": "def compute(values: list) -> int:",
}
