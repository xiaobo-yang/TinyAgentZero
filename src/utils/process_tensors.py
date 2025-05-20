import torch
from typing import Dict, List


def cut_to_effective_len(tensor_dict: Dict[str, torch.Tensor], keys: List[str], cut_left: bool = True):
    """Cut tensors to their effective length based on attention mask."""
    effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
    result = tensor_dict.copy()
    
    for key in keys:
        if cut_left:
            result[key] = tensor_dict[key][:, -effective_len:]
        else:
            result[key] = tensor_dict[key][:, :effective_len]
    return result

def create_attention_mask(input_ids: torch.Tensor, pad_token_id):
    """Create attention mask from input ids."""
    return torch.where(input_ids != pad_token_id, 1, 0)

def create_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Create position ids from attention mask."""
    return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

def concatenate_with_padding(tensors, pad_token_id, pad_to_left: bool = True):
    """Concatenate tensors and handle padding."""
    concatenated = torch.cat(tensors, dim=1)
    mask = concatenated != pad_token_id if pad_to_left else concatenated == pad_token_id
    sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
    padded_tensor = concatenated.gather(1, sorted_indices)
    return padded_tensor

def get_chat_template(tokenizer_type):
    """Add chat template to inputs."""
    tokenizer_name = tokenizer_type.lower()
    if 'qwen' in tokenizer_name:
        chat_template = {
            'start': '<|im_start|>',
            'end': '<|im_end|>',
            'user': 'user',
            'assistant': 'assistant',
            'think_start': '<think>',
            'think_end': '</think>',
        }
    elif 'llamatokenizerfast' in tokenizer_name: # Deepseek R1 distilled Qwen uses LlamaTokenizerFast
        chat_template = {
            'start': '<｜begin▁of▁sentence｜>',
            'end': '<｜end▁of▁sentence｜>',
            'user': '<｜User｜>',
            'assistant': '<｜Assistant｜>',
            'think_start': '<think>',
            'think_end': '</think>',
        }
    else:
        raise NotImplementedError(f"Chat template for {tokenizer_type} is not implemented")

    return chat_template