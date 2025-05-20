import torch
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.torch_functional import masked_mean
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """
    KL with info mask, since info tokens are not sampled from model, this break the definition of KL divergence.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    info_mask = data.batch['info_mask']
    response_mask = info_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"],
            data.batch["ref_log_prob"],
            kl_penalty=kl_penalty,
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}

    return data, metrics

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0):
    responses = data.batch["responses"]
    response_length = responses.size(-1)
    attention_mask = data.batch['attention_mask']
    info_mask = data.batch["info_mask"]
    response_mask = attention_mask[:, -response_length:]
    response_mask_with_info = info_mask[:, -response_length:]
    token_level_rewards = data.batch["token_level_rewards"]
    index = data.non_tensor_batch["uid"]

    if adv_estimator == 'gae':
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
    elif adv_estimator == 'grpo':
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, response_mask=response_mask, index=index
        )
    elif adv_estimator == 'reinforce_plus_plus':
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, response_mask=response_mask, gamma=gamma
        )
    elif adv_estimator == 'remax':
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=token_level_rewards,
            reward_baselines=reward_baselines,
            response_mask=response_mask,
        )
    elif adv_estimator == 'rloo':
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=token_level_rewards, response_mask=response_mask, index=index
        )
    else:
        raise NotImplementedError
    
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    
    return data


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

def compute_extra_sft_loss(log_prob, response_mask, success_mask):
    """
    Compute the SFT loss for the samples with the highest scores.
    """
    # Apply SFT loss to those samples (negative log likelihood)
    sft_loss, count = torch.tensor(0.0, device=log_prob.device), 0
    for idx, mask in enumerate(success_mask):
        if mask:
            sample_log_prob = log_prob[idx]
            sample_mask = response_mask[idx]
            sample_sft_loss = -verl_F.masked_mean(sample_log_prob, sample_mask)
            sft_loss += sample_sft_loss
            count += 1
    return sft_loss / len(success_mask), count

