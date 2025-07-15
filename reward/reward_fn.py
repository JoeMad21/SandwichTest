import ast
import random
from typing import List, Optional

from reward.reward_utils import (
    check_repetition,
    is_trivial_code,
    analyze_function_quality,
    analyze_code_without_function,
    calculate_length_bonus,
    check_extra_features
)

def reward_fn(code_snippets: List[str], prompts: Optional[List[str]] = None) -> List[float]:
    """
    Evaluates generated code and returns scalar rewards.

    Args:
        code_snippets: Generated Python function bodies.
        prompts: Corresponding prompts for anti-repetition checks.

    Returns:
        Reward values for each snippet.
    """
    rewards = []

    for i, code in enumerate(code_snippets):
        code = code.strip()
        reward = 0.0

        prompt = prompts[i].strip() if prompts and i < len(prompts) else None

        # Check for repetition
        is_repetitive, repetition_score = check_repetition(code, prompt)
        if is_repetitive:
            rewards.append(repetition_score)
            continue

        # Check for trivial/incomplete code
        if is_trivial_code(code):
            rewards.append(0.01)
            continue

        # Base reward for non-repetitive code
        reward += 0.1

        # Parse and analyze AST
        try:
            tree = ast.parse(code)
            reward += 0.1  # bonus for syntactic validity

            func_nodes = [n for n in ast.iter_child_nodes(tree) if isinstance(n, ast.FunctionDef)]
            
            if func_nodes:
                fn = func_nodes[0]
                reward += 0.1  # valid function structure
                
                # Analyze function quality
                quality_score = analyze_function_quality(fn, code)
                
                # Handle special negative scores
                if quality_score == -1.0:  # Useless body
                    rewards.append(0.01)
                    continue
                elif quality_score == -0.98:  # No return statement
                    rewards.append(0.02)
                    continue
                else:
                    reward += quality_score
            else:
                # No function definition
                quality_score = analyze_code_without_function(code)
                if quality_score == -0.98:
                    rewards.append(0.02)
                    continue
                reward += quality_score

        except SyntaxError:
            rewards.append(0.01)
            continue

        # Length heuristics
        lines = [line.strip() for line in code.split("\n") if line.strip()]
        reward += calculate_length_bonus(lines)

        # Extra features
        reward += check_extra_features(code)

        # Final normalization and tie-breaking
        reward = max(0.01, min(0.9, reward))
        reward += random.uniform(0.0, 0.01)
        rewards.append(round(min(reward, 0.9), 3))

    return rewards
