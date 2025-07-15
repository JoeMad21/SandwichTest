"""Example code snippets for few-shot learning or testing."""

# Good examples of summation functions
GOOD_EXAMPLES = [
    """def sum_numbers(nums):
    total = 0
    for num in nums:
        total += num
    return total""",
    
    """def sum_numbers(nums):
    return sum(nums)""",
    
    """def compute_sum(values):
    if not values:
        return 0
    result = 0
    for v in values:
        result += v
    return result""",
]

# Bad examples (for testing reward function)
BAD_EXAMPLES = [
    """def sum_numbers(nums):
    pass""",
    
    """def sum_numbers(nums):
    # TODO: implement this""",
    
    """def sum_numbers(nums):""",
    
    """def sum_numbers(nums):
    return None""",
]

# Edge cases
EDGE_CASES = [
    """def sum_numbers(nums):
    return 0""",  # Always returns 0
    
    """def sum_numbers(nums):
    print(nums)
    return sum(nums)""",  # Has side effects
    
    """def sum_numbers(nums):
    try:
        return sum(nums)
    except:
        return 0""",  # Error handling
]
