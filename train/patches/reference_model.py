import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput


class PatchedReferenceModel(nn.Module):
    """Reference model wrapper for PPO training."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        out = self.model(*args, **kwargs)
        if isinstance(out, tuple):
            return CausalLMOutput(
                logits=out[0],
                hidden_states=None,
                attentions=None,
            )
        return out
