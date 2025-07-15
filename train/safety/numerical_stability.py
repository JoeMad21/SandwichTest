import torch


def check_model_stability(model):
    """Check model weights for numerical stability issues."""
    print("🔧 Checking model weights for numerical stability...")
    weight_issues = False
    
    for name, param in model.named_parameters():
        # Check for NaN
        if torch.isnan(param).any():
            print(f"⚠️  Found NaN in parameter: {name}")
            param.data = torch.where(torch.isnan(param), torch.zeros_like(param), param)
            weight_issues = True
        
        # Check for Inf
        if torch.isinf(param).any():
            print(f"⚠️  Found Inf in parameter: {name}")
            param.data = torch.where(torch.isinf(param), torch.zeros_like(param), param)
            weight_issues = True
        
        # Check for extremely large values
        if param.abs().max() > 1e10:
            print(f"⚠️  Found very large values in parameter: {name} (max: {param.abs().max()})")
            param.data = torch.clamp(param.data, -1e10, 1e10)
            weight_issues = True
    
    if weight_issues:
        print("⚠️  Fixed numerical issues in model weights")
        return False
    
    return True


def stabilize_gradients(model, max_grad_norm=1.0):
    """Clip gradients to prevent explosion."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
