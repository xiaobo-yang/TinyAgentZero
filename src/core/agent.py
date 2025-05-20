
import os
import json
from typing import List, Dict, Literal
import torch
from verl import DataProto

from src.env import get_env
from src.utils.process_tensors import (
    cut_to_effective_len, 
    create_attention_mask, 
    create_position_ids,
    concatenate_with_padding,
    get_chat_template,
)
from src.utils.debugger import check_token_align


class Agent:
    def __init__(
        self,
        config,
        tokenizer,
        actor_rollout_wg,
        ref_policy_wg,
        is_validation: bool = False,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg # policy module
        self.ref_policy_wg = ref_policy_wg # reference policy module
        self.is_validation = is_validation # mode of agent, this may control feature of env
        self.num_gpus = config.trainer.n_gpus_per_node
        self.log_dir = config.data.log_dir # path to save trajectory of interaction
        self.max_prompt_length = config.data.max_prompt_length # truncation length of rolling prompt
        self.max_response_length = config.data.max_response_length # truncation length of response
        self.max_feedback_length = config.data.max_feedback_length # truncation length of env feedback

    def interact(self, prompts: DataProto, mode: Literal['train', 'val'], global_steps: int):
        """
        LLM interacts with environment.
        
        Input:
            prompts: data for generation.
        """
        # set mode
        self.set_mode(mode)
        # initialize env, metrics and truncated prompts
        initial_prompt_ids = self.ini_env(prompts)
        
        # prepare data
        # for prompts, we need to pad to left, while for responses, we need to pad to right
        original_prompt = {"input_ids": initial_prompt_ids}
        responses = {
            'responses': initial_prompt_ids[:, []],
            'responses_with_info_mask': initial_prompt_ids[:, []],
            'responses_reward_tensors': initial_prompt_ids[:, []],
        }

        # interaction loop
        for turn in range(self.max_turns):
            self.turn = turn
            # break if all task are finished
            if self.all_finished():
                break
            
            # generate raw responses, which may be filtered out by env
            raw_responses_str = self.generate(prompts)

            # send actions to env, receive feedback and filtered responses
            new_responses_ids, feedbacks_ids = self.send_action(raw_responses_str, prompts)

            # append the new responses and feedbacks to the old prompts to create new prompts for the next turn.
            prompts, responses = self.append_info(prompts, responses, new_responses_ids, feedbacks_ids)

        
        # final output = original prompts + responses
        output_batch = self.compose_final_output(original_prompt, responses)

        # save trajectory
        self.save_trajectory(global_steps)

        return output_batch

    def ini_env(self, prompts: DataProto):
        """
        Re-initialize the environments and metrics, truncate prompts to max_prompt_length.
        """
        # truncate prompts and add them to metrics
        initial_prompt_ids = prompts.batch["input_ids"][:, -self.max_prompt_length :] # NOTE: without truncation, GPU may OOM
        initial_prompt_str = self.tokenizer.batch_decode(initial_prompt_ids, skip_special_tokens=True)
        B, T = prompts.batch["input_ids"].shape
        env_trajs = {'initial_prompt': {idx: None for idx in range(B)}}
        for idx in range(B):
            env_trajs['initial_prompt'][idx] = initial_prompt_str[idx]

        # initialize env and metrics
        split = 'val' if self.is_validation else 'train'
        self.env = get_env(self.config, n_envs=B, split=split, paraphrase=self.config.algorithm.paraphrase.enable)
        self.metrics = {
            "active_mask": torch.ones(B, dtype=torch.bool),
            "active_num_list": [B,],
            "env_trajs": env_trajs, 
            "all_step_rewards": [],
            }
        max_turns = self.config.data.val_max_turns if self.is_validation else self.config.data.max_turns
        self.max_turns = max_turns

        return initial_prompt_ids


    def all_finished(self):
        """
        All task are finished.
        """
        active_mask = self.metrics["active_mask"]
        return active_mask.sum() == 0

    def generate(self, prompts: DataProto, use_ref_policy: bool = False):
        """
        Generate action for environment.
        """
        # remove redundant left padding tokens
        prompts.batch = cut_to_effective_len(
            prompts.batch, keys=["input_ids", "attention_mask", "position_ids"]
        )

        # filter out finished interaction
        active_mask = self.metrics["active_mask"]

        prompts = DataProto.from_dict(
                    tensors={k: v[active_mask] for k, v in prompts.batch.items()},
                    non_tensors={k: v[active_mask] for k, v in prompts.non_tensor_batch.items()},
                    meta_info=prompts.meta_info,
                )

        # generate
        # NOTE: HF tokenizer will output fp32 tensor for empty string, 
        # that will make new prompt tokens `cat([prompt, response, feedbacks])` become fp32
        prompts.batch["input_ids"] = prompts.batch["input_ids"].long()  
        prompts.batch["attention_mask"] = prompts.batch["attention_mask"].long()
        prompts.batch["position_ids"] = prompts.batch["position_ids"].long()
        
        responses = self._generate_with_gpu_padding(prompts, use_ref_policy)
        
        responses_ids = responses.batch["responses"].long()
        responses_str = self.tokenizer.batch_decode(responses_ids, skip_special_tokens=False) # keep special tokens, otherwise the response will be empty
        responses_str = [resp.split(self.tokenizer.eos_token)[0] for resp in responses_str] # remove eos token
        
        
        # make responses have same batch size with original prompts
        responses_str = self._pad_inactive_samples(responses_str)


        return responses_str


    def send_action(self, responses_str: List[str], prompts: DataProto):
        """
        Send actions to env and receive feedback. Batch contains some information that may be used in env.
        """
        # use chat template in feedback
        chat_template = get_chat_template(self.tokenizer.__class__.__name__)

        # send all task instructions to the env, then get the feedbacks
        # and some texts from responses may be filtered out by env
        filtered_responses_str, feedbacks, dones, step_rewards, env_trajs = self.env.run(responses_str, prompts, chat_template)
        new_responses_ids = self.tokenizer(filtered_responses_str, padding='longest', return_tensors='pt', add_special_tokens=False)['input_ids']
        feedbacks_ids = self._process_feedbacks(feedbacks)
        
        # update active samples
        self.metrics["active_mask"] = torch.tensor([not done for done in dones], dtype=torch.bool)
        self.metrics["active_num_list"].append(self.metrics["active_mask"].sum().item())
        self.metrics["env_trajs"].update(env_trajs)
        self.metrics["all_step_rewards"].append(step_rewards)

        # rephrase the feedbacks
        if self.config.algorithm.paraphrase.enable:
            feedbacks_ids = self.paraphrase(prompts, new_responses_ids, feedbacks_ids)

        return new_responses_ids, feedbacks_ids

    def mcts_search(self, prompts: DataProto):
        """
        MCTS search.
        """
        chat_template = get_chat_template(self.tokenizer.__class__.__name__)
        B, T = prompts.batch["input_ids"].shape

        choices = {i: [] for i in range(B)} # choices for each prompt
        for rep in range(self.config.algorithm.mcts_n):
            if (self.metrics["active_mask"] == False).all():
                break
            # generate raw responses, which may be filtered out by env
            raw_responses_str = self.generate(prompts)

            # send all task instructions to the env, then get the feedbacks
            # and some texts from responses may be filtered out by env
            filtered_responses_str, feedbacks, dones, step_rewards, env_trajs = self.env.run(raw_responses_str, prompts, chat_template)
            # finished samples do not need to be considered in the next rep
            self.metrics["active_mask"] = torch.tensor([not done for done in dones], dtype=torch.bool)

            for i in range(B):
                choices[i].append((filtered_responses_str[i], feedbacks[i], step_rewards[i]))

        # select the best choice
        filtered_responses_str, feedbacks, step_rewards = [], [], []
        for i in range(B):
            # select maximum step reward choice # TODO: consider other metrics for same reward, e.g. length
            best_choice = max(choices[i], key=lambda x: x[2])
            filtered_resp_str, feedback, step_reward = best_choice
            
            # reconstruct the metrics
            filtered_responses_str.append(filtered_resp_str)
            feedbacks.append(feedback)
            step_rewards.append(step_reward)

        new_responses_ids = self.tokenizer(filtered_responses_str, padding='longest', return_tensors='pt', add_special_tokens=False)['input_ids']
        feedbacks_ids = self._process_feedbacks(feedbacks)
        
        # update active samples
        self.metrics["active_num_list"].append(self.metrics["active_mask"].sum().item())
        self.metrics["all_step_rewards"].append(step_rewards)
        # TODO: env_trajs save all mcts action sequentially, e.g. [a1, a1, a1, a2, a2, a2, a3, a3, a3, ...]
        # may use mcts_n to add a postprocess fn in save_trajectory
        self.metrics["env_trajs"] = env_trajs 

        return new_responses_ids, feedbacks_ids

    def paraphrase(self, prompts: DataProto, new_responses_ids: torch.Tensor, feedbacks_ids: torch.Tensor):
        """
        Rephrase the feedbacks. The policy first generate a reflection cot on response_str, 
        then use the cot as new thinking cot in agent
        """
        if  self.turn == self.max_turns - 1:
            for idx in range(prompts.batch["input_ids"].shape[0]):
                self.metrics["env_trajs"]['trajectory'][idx][-1]['feedback'] = ''
                self.metrics["env_trajs"]['trajectory'][idx][-1]['raw_feedback'] = ''
            return feedbacks_ids
        
        chat_template = get_chat_template(self.tokenizer.__class__.__name__)

        # use repharse instruction, generate some rephrased feedbacks
        prompts_for_paraphrase = self._append_prompt(prompts, new_responses_ids, feedbacks_ids)
        if self.config.algorithm.paraphrase.use_ref_policy:
            use_ref_policy = True
        else:
            use_ref_policy = False
        paraphrased_feedback_strs = self.generate(prompts_for_paraphrase, use_ref_policy=use_ref_policy)
            
        
        new_feedback_strs = []
        for i, paraphrased_feedback_str in enumerate(paraphrased_feedback_strs):
            # log raw rephrased feedback
            self.metrics["env_trajs"]['trajectory'][i][-1]['raw_feedback'] = paraphrased_feedback_str
            if not self.metrics["active_mask"][i]:
                new_feedback_strs.append("")
                continue

            new_feedback_str = ""
            if chat_template['end'] not in paraphrased_feedback_str:
                new_feedback_str += f"{chat_template['end']}"

            # postprocess the feedback
            tmp_feedback_str = paraphrased_feedback_str.replace(chat_template['think_start'], "").replace(chat_template['think_end'], "")
            tmp_feedback_str = tmp_feedback_str.split("<answer>")[0]
            
            # use rephrased feedback as new thinking cot
            if tmp_feedback_str:
                new_feedback_str = f"\n{chat_template['think_start']}\n{tmp_feedback_str}{chat_template['think_end']}\n{chat_template['think_start']}"
            else:
                new_feedback_str = f"\n{chat_template['think_start']}"                            
            new_feedback_strs.append(new_feedback_str)
        new_feedbacks_ids = self._process_feedbacks(new_feedback_strs)
            
        # replace old feedback with rephrased feedback
        for idx, new_feedback in enumerate(new_feedback_strs):
            self.metrics["env_trajs"]['trajectory'][idx][-1]['feedback'] = new_feedback
        
        return new_feedbacks_ids


    def append_info(self, prompts, responses, new_responses_ids, feedbacks_ids):
        """
        Append new information to old response, as a new prompt for next turn.
        """
        if self.turn < self.max_turns - 1:
            prompts = self._append_prompt(prompts, new_responses_ids, feedbacks_ids)
            responses = self._append_response(responses, new_responses_ids, feedbacks_ids)
        else:
            responses = self._append_response(responses, new_responses_ids, None)
        
        return prompts, responses

    def compose_final_output(self, left_side: Dict, right_side: Dict):
        """
        Compose final generation output, including responses, masks and rewards.
        """
        final_output = right_side.copy()
        final_output["prompts"] = left_side["input_ids"]

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)

        # Create attention mask, info(from env) mask and position ids
        pad_token_id = self.tokenizer.pad_token_id
        final_output['attention_mask'] = torch.cat([
            create_attention_mask(left_side['input_ids'], pad_token_id),
            create_attention_mask(final_output['responses'], pad_token_id)
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            create_attention_mask(left_side['input_ids'], pad_token_id),
            create_attention_mask(final_output['responses_with_info_mask'], pad_token_id)
        ], dim=1)
        final_output['position_ids'] = create_position_ids(
            final_output['attention_mask']
        )

        # reallocate reward tensor
        reward_tensors = right_side['responses_reward_tensors']
        reward_tensors = self._reallocate_step_rewards(reward_tensors)
        final_output["token_level_scores"] = reward_tensors
        final_output["sample_level_scores"] = reward_tensors.sum(dim=-1)
        

        print("ACTIVE_TRAJ_NUM:", self.metrics["active_num_list"])
        final_output = DataProto.from_dict(
            final_output, meta_info={"active_num_list": self.metrics["active_num_list"]}
        )
        # avoid fp32 dtype bug in tokens tensor
        final_output.batch["input_ids"] = final_output.batch["input_ids"].long()  
        final_output.batch["attention_mask"] = final_output.batch["attention_mask"].long()
        final_output.batch["position_ids"] = final_output.batch["position_ids"].long()
        final_output.batch["responses"] = final_output.batch["responses"].long()
        final_output.batch["info_mask"] = final_output.batch["info_mask"].long()

        # add success mask
        final_output.batch["success_mask"] = (~self.metrics["active_mask"]).long()

        if self.config.debug.check_token_align:
            check_token_align(final_output, self.tokenizer)

        return final_output

    def save_trajectory(self, global_steps):
        """
        Save the entire interaction trajectory of the agent.
        """
        split = 'val' if self.is_validation else "train"
        output_dir = f'{self.log_dir}/step_{global_steps}'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f'{output_dir}/{split}.jsonl', 'w', encoding='utf-8') as f:
            env_trajs = self.metrics["env_trajs"]
            json_str = json.dumps(env_trajs, indent=4)
            f.write(json_str + '\n')

    def set_mode(self, mode: Literal['val', 'train']):
        """
        Set mode of agent.
        """
        if mode == 'val':
            self.is_validation = True
        else:
            self.is_validation = False

    def _process_feedbacks(self, feedbacks: List[str]) -> torch.Tensor:
        """
        Process next observations from environment.
        """

        feedbacks_ids = self.tokenizer(
            feedbacks, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if feedbacks_ids.shape[1] > self.max_feedback_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {feedbacks_ids.shape[1]} & {self.max_feedback_length}") 
            feedbacks_ids = feedbacks_ids[:, : self.max_feedback_length]

        return feedbacks_ids

    def _append_prompt(self, prompts: DataProto, new_responses_ids: torch.Tensor, feedbacks_ids: torch.Tensor) -> Dict:
        """
        Update rolling state with new responses and observations.
        """
        # Concatenate and handle padding
        pad_token_id = self.tokenizer.pad_token_id
        new_input_ids = concatenate_with_padding(
            [prompts.batch["input_ids"], new_responses_ids, feedbacks_ids],
            pad_token_id,
        )

        # Create attention mask and position ids
        new_attention_mask = create_attention_mask(new_input_ids, pad_token_id)
        new_position_ids = create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.max_prompt_length + (self.max_response_length + self.max_feedback_length) * self.turn, effective_len)

        new_rollings = DataProto.from_dict(
            tensors={
                'input_ids': new_input_ids[:, -max_len:],
                'position_ids': new_position_ids[:, -max_len:],
                'attention_mask': new_attention_mask[:, -max_len:]
            }, 
            non_tensors=prompts.non_tensor_batch, 
            meta_info=prompts.meta_info,
        )
        
        return new_rollings

    def _append_response(self, responses_batch: Dict, new_responses_ids, feedbacks_ids):
        """
        Update right side state.
        """
        if feedbacks_ids != None:
            responses, responses_with_info_mask, responses_reward_tensors = self._info_masked_concatenate_with_padding(
                    responses_batch['responses'],
                    responses_batch['responses_with_info_mask'],
                    responses_batch['responses_reward_tensors'],
                    new_responses_ids,
                    feedbacks_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask, responses_reward_tensors = self._info_masked_concatenate_with_padding(
                    responses_batch['responses'],
                    responses_batch['responses_with_info_mask'],
                    responses_batch['responses_reward_tensors'],
                    new_responses_ids,
                    pad_to_left=False
                )
        effective_len = create_attention_mask(responses, self.tokenizer.pad_token_id).sum(dim=1).max()
        
        return {'responses': responses[:, :effective_len], 
                'responses_with_info_mask': responses_with_info_mask[:, :effective_len],
                'responses_reward_tensors': responses_reward_tensors[:, :effective_len],
                }

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                prompt_reward_tensor: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """
        Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists.
        Also, create aligned process reward.
        """
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        # construct step reward
        step_tensors_reward = torch.zeros_like(response, dtype=torch.float32)
        for idx in range(len(response)):
            valid_response_length = (response[idx] != pad_id).sum()
            if valid_response_length > 0:
                step_tensors_reward[idx, valid_response_length - 1] = 1.0 # just for marking the end of the response
        tensors_reward = [prompt_reward_tensor, step_tensors_reward]

        # information from env
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
            tensors_reward.append(torch.zeros_like(info, dtype=torch.float32))
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        concatenated_tensor = torch.cat(tensors_reward, dim=1)
        
        # move pad tokens to one side
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        
        # for info and reward, we need to use the same padding method to ensure alignment
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        padded_reward_tensor = concatenated_tensor.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info, padded_reward_tensor

    def _generate_with_gpu_padding(self, active_batch: DataProto, use_ref_policy: bool = False) -> DataProto:
        """
        Wrapper for generation that handles multi-GPU padding requirements.
        if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
        if active_batch size is not divisible by num_gpus, pad with first sequence
        then remove padding from output
        """
        num_gpus = self.num_gpus
        if num_gpus <= 1:
            if use_ref_policy:
                return self.ref_policy_wg.generate_sequences(active_batch)
            else:
                return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            if use_ref_policy:
                return self.ref_policy_wg.generate_sequences(active_batch)
            else:
                return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1)) # (1, 1, 1, ...) -> (padding_size, 1, 1, ...)
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(
            tensors=padded_batch,
            meta_info=active_batch.meta_info,  # include some meta info for generation, e.g. eos_token_id, pad_token_id, do_sample, etc.
            )

        # Generate with padded batch
        if use_ref_policy:
            padded_output = self.ref_policy_wg.generate_sequences(padded_active_batch)
        else:
            padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output
    
    def _pad_inactive_samples(self, responses_str: List[str]):
        """
        Pad responses for non-active examples with empty strings.
        """
        active_mask = self.metrics["active_mask"]
        assert active_mask.sum() == len(responses_str), f"active_mask.sum() = {active_mask.sum()}, len(responses_str) = {len(responses_str)}"
        batch_size = active_mask.shape[0]
        # Create masked response strings
        padded_responses_str = [""] * batch_size
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1
                
        return padded_responses_str
    
    def _reallocate_step_rewards(self, reward_tensors: torch.Tensor):
        all_step_rewards = self.metrics["all_step_rewards"] # List[step_rewards]
        all_step_rewards = torch.tensor(all_step_rewards).transpose(0, 1)  # (n_samples, n_turns)
        reward_tensors = reward_tensors.clone()
        batch_size = reward_tensors.shape[0]
        method = self.config.data.reward_allocate_method
        # method to allocate rewards:
        #     'step' is to allocate step rewards to pre-defined positions, 
        #     'sum' is to allocate sum of all step rewards to last pre-defined positions, 
        #     'from_env' is to allocate rewards from env. 
        
        if method == 'step':
            for idx in range(batch_size):
                # find pre-defined reward positions
                marked_positions = (reward_tensors[idx] == 1.0).nonzero(as_tuple=True)[0]
                for turn, pos in enumerate(marked_positions):
                    reward_tensors[idx, pos] = all_step_rewards[idx, turn]
        
        elif method == 'sum':
            for idx in range(batch_size):
                marked_positions = (reward_tensors[idx] == 1.0).nonzero(as_tuple=True)[0]
                total_reward = all_step_rewards[idx].sum()
                reward_tensors[idx] = 0.0 # remove pre-defined markers
                reward_tensors[idx, marked_positions[-1]] = total_reward
        
        elif method == 'from_env':
            if not hasattr(self.env, 'get_reward_allocation'):
                raise ValueError(f"Env class '{self.env.__class__.__name__}' does not provide 'get_reward_allocation()' method for reward allocation")
            else:
                reward_tensors = self.env.get_reward_allocation(reward_tensors)
        else:
            raise NotImplementedError(f"Reward allocation method '{method}' is not supported")

        return reward_tensors