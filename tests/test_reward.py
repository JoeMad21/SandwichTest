"""Tests for reward function and utilities."""

import pytest
from reward.reward_fn import reward_fn
from reward.reward_utils import check_repetition, is_trivial_code, analyze_function_quality


class TestRewardFunction:
    """Test the main reward function."""
    
    def test_valid_sum_function(self):
        """Test reward for valid sum function."""
        code = """def sum_numbers(nums):
    total = 0
    for num in nums:
        total += num
    return total"""
        
        rewards = reward_fn([code])
        assert len(rewards) == 1
        assert rewards[0] > 0.5  # Should get high reward
    
    def test_trivial_function(self):
        """Test reward for trivial function."""
        code = "def sum_numbers(nums): pass"
        rewards = reward_fn([code])
        assert len(rewards) == 1
        assert rewards[0] < 0.02  # Should get very low reward
    
    def test_syntax_error(self):
        """Test reward for code with syntax error."""
        code = "def sum_numbers(nums) total = 0"  # Missing colon
        rewards = reward_fn([code])
        assert len(rewards) == 1
        assert rewards[0] == 0.01  # Should get minimum reward
    
    def test_repetitive_code(self):
        """Test reward for repetitive code."""
        prompt = "def sum_numbers(nums):"
        code = "def sum_numbers(nums):"  # Exact repetition
        rewards = reward_fn([code], [prompt])
        assert len(rewards) == 1
        assert rewards[0] == 0.001  # Should get repetition penalty
    
    def test_multiple_snippets(self):
        """Test reward for multiple code snippets."""
        codes = [
            "def sum_numbers(nums): return sum(nums)",
            "def sum_numbers(nums): pass",
            "invalid python code",
        ]
        rewards = reward_fn(codes)
        assert len(rewards) == 3
        assert rewards[0] > rewards[1] > rewards[2]


class TestRewardUtils:
    """Test reward utility functions."""
    
    def test_check_repetition(self):
        """Test repetition checking."""
        # Exact match
        is_rep, score = check_repetition("def foo():", "def foo():")
        assert is_rep is True
        assert score == 0.001
        
        # No repetition
        is_rep, score = check_repetition("def foo(): return 1", "def foo():")
        assert is_rep is False
        assert score == 0.0
        
        # High overlap
        is_rep, score = check_repetition("def foo(): ", "def foo():")
        assert is_rep is True
    
    def test_is_trivial_code(self):
        """Test trivial code detection."""
        assert is_trivial_code("def foo(): pass") is True
        assert is_trivial_code("def foo(): ...") is True
        assert is_trivial_code("def foo(): # TODO") is True
        assert is_trivial_code("def foo():") is True
        assert is_trivial_code("def foo(): return 1") is False
    
    def test_analyze_function_quality(self):
        """Test function quality analysis."""
        import ast
        
        # Good function
        code = """def sum_numbers(nums):
    total = 0
    for n in nums:
        total += n
    return total"""
        tree = ast.parse(code)
        func = tree.body[0]
        score = analyze_function_quality(func, code)
        assert score > 0.2
        
        # Function without return
        code = """def sum_numbers(nums):
    total = 0"""
        tree = ast.parse(code)
        func = tree.body[0]
        score = analyze_function_quality(func, code)
        assert score == -0.98  # Special score for no return


@pytest.fixture
def sample_codes():
    """Fixture providing sample code snippets."""
    return {
        "good": """def sum_numbers(nums):
    return sum(nums)""",
        "bad": "def sum_numbers(nums): pass",
        "syntax_error": "def sum_numbers(nums",
        "empty": "",
    }
