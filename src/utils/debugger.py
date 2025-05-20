import random
from verl import DataProto

def check_token_align(batch: DataProto, tokenizer, samle_idx=None):
    """
    Pretty print tokens, attention mask, info mask, and reward information for a single sample.
    
    Args:
        batch: DataProto batch
        tokenizer: The tokenizer to decode token IDs
    """
    responses = batch.batch['responses']
    
    # sample random index from batch
    sample_idx = random.randint(0, responses.shape[0]-1) if samle_idx is None else samle_idx
    response_tokens = tokenizer.convert_ids_to_tokens(batch.batch['responses'][sample_idx])
    reward_tensor = batch.batch['token_level_scores'][sample_idx]
    attention_mask = batch.batch['attention_mask'][sample_idx, -responses.shape[1]:] # remove tokens from prompts
    info_mask = batch.batch['info_mask'][sample_idx, -responses.shape[1]:]
 
    print("len(response_tokens):", len(response_tokens))
    print("len(attention_mask):", len(attention_mask))
    print("len(info_mask):", len(info_mask))
    print("len(reward_tensor):", len(reward_tensor))
    print("total reward:", reward_tensor.sum().item())
    
    # Create table header
    print("\nToken-by-token breakdown:")
    print("-" * 120)
    print(f"{'Index':>6} | {'Token':<40} | {'Attention':>8} | {'Info Mask':>10} | {'Reward':>10}")
    print("-" * 120)
    
    # Print response tokens with rewards
    print("RESPONSE TOKENS:")
    for idx, token in enumerate(response_tokens):
        # Clean up token display
        display_token = token.replace('Ġ', ' ').replace('▁', ' ')
        
        attn = attention_mask[idx]
        info = info_mask[idx]
        reward = f"{reward_tensor[idx].item():.4f}"
            
        print(f"{idx:>6} | {display_token:<40} | {int(attn):>8} | {int(info):>10} | {reward:>10}")
    
    print("-" * 120) 