"""Tests for generation utilities."""

import pytest
import torch
from train.generation_utils import clean_prompt, clean_code, safe_generate


class TestGenerationUtils:
    """Test generation utility functions."""
    
    def test_clean_prompt(self):
        """Test prompt cleaning."""
        # Test removal of instruction phrases
        prompt = "Complete the Python function:\ndef foo():"
        cleaned = clean_prompt(prompt)
        assert cleaned == "def foo():"
        
        # Test already clean prompt
        prompt = "def bar():"
        cleaned = clean_prompt(prompt)
        assert cleaned == "def bar():"
        
        # Test whitespace handling
        prompt = "  def baz():  "
        cleaned = clean_prompt(prompt)
        assert cleaned == "def baz():"
    
    def test_clean_code(self):
        """Test code cleaning."""
        prompt = "def foo():"
        
        # Test removal of prompt prefix
        code = "def foo(): return 1"
        cleaned = clean_code(code, prompt)
        assert cleaned == "return 1"
        
        # Test comment removal
        code = """def foo():
    # This is a comment
    return 1"""
        cleaned = clean_code(code, prompt)
        assert "# This is a comment" not in cleaned
        assert "return 1" in cleaned
        
        # Test empty line handling
        code = """def foo():
    
    return 1
    
    """
        cleaned = clean_code(code, prompt)
        assert cleaned.strip() == "return 1"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_safe_generate(self):
        """Test safe generation (requires CUDA)."""
        # This test would require a real model and tokenizer
        # Marking as skip for now
        pass


class TestGenerationPatches:
    """Test generation patches."""
    
    def test_patch_imports(self):
        """Test that patch modules can be imported."""
        from train.patches.generation_patches import patch_generation_methods
        from train.patches.trl_patches import patch_trl_batch_generation
        
        # Just verify imports work
        assert callable(patch_generation_methods)
        assert callable(patch_trl_batch_generation)
