import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import kl_penalty, agg_loss
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
import verl.utils.torch_functional as verl_F

from src.utils.algorithm import compute_policy_loss, compute_extra_sft_loss


def update_policy_with_info_mask(self, data: DataProto):
    # make sure we are in training mode
    self.actor_module.train()

    temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

    select_keys = ['responses', 'input_ids', 'attention_mask', 'info_mask', 'position_ids', 'old_log_probs', 'advantages']  # NOTE: use info_mask to mask out the env feedback tokens
    if self.config.use_kl_loss:
        select_keys.append('ref_log_prob')
    sft_weight = self.config.sft_weight
    if sft_weight > 0:
        select_keys.append("success_mask")
        all_success_mask = data.batch["success_mask"]
    
    batch = data.select(batch_keys=select_keys).batch
    has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

    # Split to make minibatch iterator for updating the actor
    # See PPO paper for details. https://arxiv.org/abs/1707.06347
    if has_multi_modal_inputs:
        num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
        non_tensor_select_keys = ['multi_modal_inputs']
        dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
    else:
        dataloader = batch.split(self.config.ppo_mini_batch_size)

    metrics = {}
    for epoch in range(self.config.ppo_epochs):
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if has_multi_modal_inputs:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
            elif self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                # Support all hardwares
                if isinstance(data, DataProto):
                    data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                else:
                    data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                info_mask = data['info_mask']
                response_mask = info_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                # all return: (bsz, response_length)
                calculate_entropy = False
                if entropy_coeff != 0:
                    calculate_entropy = True
                entropy, log_prob = self._forward_micro_batch(
                    micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy
                )

                # policy loss
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    cliprange=clip_ratio,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode=loss_agg_mode,
                )

                # sft loss, only apply to the samples with the higher scores
                if sft_weight > 0 and all_success_mask.sum() > 0:
                    # for sft, we need to learn environment feedback
                    attention_mask = data['attention_mask'][:, -response_length:]
                    success_mask = data["success_mask"]
                    sft_loss, success_count = compute_extra_sft_loss(log_prob, attention_mask, success_mask)
                    if success_count > 0:
                        pg_loss = pg_loss + sft_weight * sft_loss
                else:
                    sft_loss = torch.tensor(0.0, device=pg_loss.device)
                    success_count = 0
                
                # compute entropy loss from entropy
                if entropy_coeff != 0:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff
                else:
                    policy_loss = pg_loss
                    entropy_loss = torch.tensor(0.0)

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = kl_penalty(logprob=log_prob,
                                        ref_logprob=ref_log_prob,
                                        kl_penalty=self.config.kl_loss_type)
                    kl_loss = agg_loss(loss_mat=kld,
                                        loss_mask=response_mask,
                                        loss_agg_mode=self.config.loss_agg_mode)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                    'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item(),
                }
                if sft_weight > 0:
                    data.update({
                        'actor/extra_sft_loss': sft_loss.detach().item(),
                        'actor/success_count': success_count,
                    })
                append_to_dict(metrics, data)

            
            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
        append_to_dict(metrics, data)
    self.actor_optimizer.zero_grad()
    return metrics
