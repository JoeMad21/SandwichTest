import ast
from typing import List, Tuple, Optional

def check_repetition(code: str, prompt: Optional[str]) -> Tuple[bool, float]:
    """Check if code is repetitive of prompt, return (is_repetitive, penalty_score)"""
    code = code.strip()
    prompt = prompt.strip() if prompt else None
    
    if not prompt:
        return False, 0.0
    
    if code == prompt:
        return True, 0.001
    
    if code.startswith(prompt) and len(code[len(prompt):].strip()) < 10:
        return True, 0.005
    
    overlap = sum(1 for c in prompt if c in code) / len(prompt)
    if overlap > 0.9 and len(code) <= 1.2 * len(prompt):
        return True, 0.01
    
    return False, 0.0

def is_trivial_code(code: str) -> bool:
    """Check if code is trivial (empty, TODO, just pass, etc)"""
    if any(code.endswith(suffix) for suffix in [":", "...", "# TODO", "pass"]):
        return True
    
    lines = [line.strip() for line in code.split("\n") if line.strip()]
    if len(lines) == 1 and lines[0].startswith("def ") and lines[0].endswith(":"):
        return True
    
    return False

def analyze_function_quality(func_node: ast.FunctionDef, code: str) -> float:
    """Analyze AST function node and return quality score"""
    score = 0.0
    
    # Empty or useless body check
    body = func_node.body
    useful = any(
        not isinstance(node, (ast.Pass, ast.Expr))
        or not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Str))
        for node in body
    )
    if not useful:
        return -1.0  # Signal to parent to give minimal reward
    
    # Check for return statement
    if any(isinstance(n, ast.Return) for n in ast.walk(func_node)):
        score += 0.15
    else:
        return -0.98  # Signal very low reward for no return
    
    # Check parameter usage
    if func_node.args.args:
        param = func_node.args.args[0].arg
        used = any(
            isinstance(n, ast.Name) and n.id == param
            for n in ast.walk(func_node)
            if not isinstance(n, ast.arg)
        )
        score += 0.1 if used else -0.02
    
    # Check complexity
    stmt_count = len([
        n for n in ast.walk(func_node)
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.Return, ast.For, ast.While))
    ])
    if stmt_count <= 2:
        score -= 0.02
    else:
        score += 0.05
    
    # Check function name quality
    good_names = {"add_numbers", "sum_list", "sum_array", "calculate_total", "compute_sum"}
    if func_node.name in good_names:
        score += 0.1
    elif any(x in func_node.name.lower() for x in ["sum", "add", "total", "calc"]):
        score += 0.05
    
    # Check for summation patterns
    lower_code = code.lower()
    if any(keyword in lower_code for keyword in ["sum(", "+=", "total", "accumulate"]):
        score += 0.15
        if "sum(" in lower_code:
            score += 0.05
    
    return score

def analyze_code_without_function(code: str) -> float:
    """Analyze code that doesn't define a function"""
    if not any(kw in code.lower() for kw in ["sum(", "range(", "return", "="]):
        return -0.98  # Very low score
    return 0.1

def calculate_length_bonus(code_lines: List[str]) -> float:
    """Calculate bonus/penalty based on code length"""
    num_lines = len(code_lines)
    if 2 <= num_lines <= 8:
        return 0.05
    elif num_lines > 15:
        return -0.05
    return 0.0

def check_extra_features(code: str) -> float:
    """Check for advanced features like imports, try/except"""
    if any(kw in code for kw in ["import ", "try:", "except:"]):
        return 0.05
    return 0.0
