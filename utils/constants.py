import torch
import torch.nn as nn

class ValueHeadMixin:
    """
    Minimal TRL-compatible value head. Exposes `.v_head` and `.score()`.
    """
    def _init_value_head(self):
        hidden_size = self.config.hidden_size
        dtype = next(self.parameters()).dtype  # Match model dtype
        self.v_head = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False).to(dtype)
        )

    def score(self, hidden_states):
        """
        Apply the value head to hidden states.
        Returns (batch_size,) scalar value per sequence.
        """
        return self.v_head(hidden_states).squeeze(-1)


def add_value_head(model, tokenizer_vocab_size=None):
    """
    Dynamically attaches the value head to any Hugging Face model.
    Also ensures ALL embedding layers match the tokenizer vocab size.
    """
    if not isinstance(model, ValueHeadMixin):
        model.__class__ = type(
            model.__class__.__name__ + "WithValueHead",
            (ValueHeadMixin, model.__class__),
            {}
        )
    
    model._init_value_head()
    
    # FIXED: Resize ALL vocab-related layers to match tokenizer
    if tokenizer_vocab_size is not None:
        current_vocab_size = model.config.vocab_size
        if current_vocab_size != tokenizer_vocab_size:
            print(f"🔧 Resizing model vocabulary: {current_vocab_size} -> {tokenizer_vocab_size}")
            
            # Method 1: Use the built-in resize method (handles all layers)
            try:
                model.resize_token_embeddings(tokenizer_vocab_size)
                print(f"✅ Successfully resized token embeddings using built-in method")
            except Exception as e:
                print(f"⚠️  Built-in resize failed ({e}), doing manual resize...")
                
                # Method 2: Manual resize if built-in fails
                # Resize input embeddings
                if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    old_embeddings = model.model.embed_tokens
                    new_embeddings = nn.Embedding(
                        tokenizer_vocab_size, 
                        old_embeddings.embedding_dim,
                        padding_idx=old_embeddings.padding_idx,
                        dtype=old_embeddings.weight.dtype,
                        device=old_embeddings.weight.device
                    )
                    
                    # Copy existing embeddings
                    min_vocab_size = min(current_vocab_size, tokenizer_vocab_size)
                    new_embeddings.weight.data[:min_vocab_size] = old_embeddings.weight.data[:min_vocab_size]
                    model.model.embed_tokens = new_embeddings
                    print(f"✅ Resized input embeddings")
                
                # Resize output head (language model head)
                if hasattr(model, 'lm_head'):
                    old_lm_head = model.lm_head
                    new_lm_head = nn.Linear(
                        old_lm_head.in_features, 
                        tokenizer_vocab_size, 
                        bias=old_lm_head.bias is not None,
                        dtype=old_lm_head.weight.dtype,
                        device=old_lm_head.weight.device
                    )
                    
                    # Copy weights for existing vocab
                    min_vocab_size = min(current_vocab_size, tokenizer_vocab_size)
                    new_lm_head.weight.data[:min_vocab_size] = old_lm_head.weight.data[:min_vocab_size]
                    
                    if old_lm_head.bias is not None:
                        new_lm_head.bias.data[:min_vocab_size] = old_lm_head.bias.data[:min_vocab_size]
                    
                    model.lm_head = new_lm_head
                    print(f"✅ Resized output head")
            
            # Update config
            model.config.vocab_size = tokenizer_vocab_size
            print(f"✅ Updated model config vocab_size to {tokenizer_vocab_size}")
    
    return model
