import uuid
from pprint import pprint
import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import _timer

from src.patch.trainer_patch import PatchedRayPPOTrainer
from src.core.agent import Agent
from src.utils.algorithm import (
    apply_kl_penalty,
    compute_advantage,
)
from src.utils.process_metrics import (
    reduce_metrics,
    compute_data_metrics,
    compute_timing_metrics,
    compute_throughout_metrics,
    collect_active_num_list,
)


# ----------------------------------------------------------------
# ----------------------- Main Agent Trainer ---------------------
# ----------------------------------------------------------------

class RayAgentTrainer(PatchedRayPPOTrainer):

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
    ):
        super().__init__(
                config,
                tokenizer,
                role_worker_mapping,
                resource_pool_manager,
                ray_worker_group_cls,
                processor,
            )
        # get module
        self.init_workers()
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # get agent
        self.config = config
        self.agent = Agent(
            config=self.config,
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            ref_policy_wg=self.ref_policy_wg,
        )

        if self.config.algorithm.adv_estimator == 'remax':
            raise NotImplementedError
    

    def _prepare_data(self, data: dict, split: str):
        """
        Transform data dict into DataProto
        """
        # get DataProto
        batch = DataProto.from_single_dict(data)
        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object) # unique id, grpo will use this to align the reward
        repeat_times = self.config.actor_rollout_ref.rollout.n_agent if split == 'train' else self.config.actor_rollout_ref.rollout.val_kwargs.n
        batch = batch.repeat(
            repeat_times=repeat_times,
            interleave=True,
        )      
        # pop those keys for generation
        gen_batch = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "extra_info"],
        )

        return batch, gen_batch


    def _validate(self):
        assert not self.config.reward_model.enable, "not implemented for model based reward"

        test_scores, test_all_active_num_lists = [], []
        for test_data in self.val_dataloader: # NOTE: verl may use all data togather, only 1 batch
            timing_raw = {}
            # # pad data to be multiples of world size for DP
            # len_test_batch = test_data['input_ids'].shape[0]
            # pad_len_test_batch = len_test_batch - (len_test_batch % self.actor_rollout_wg.world_size)
            # test_data = {k: v[:pad_len_test_batch] for k, v in test_data.items()}
            # prepare DataProto
            test_batch, test_gen_batch = self._prepare_data(test_data, split='val')

            # for validation, we don't need to do sample
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            with _timer("step", timing_raw):
                # agent set to val for ini val env
                with _timer("gen", timing_raw):
                    test_gen_batch_output = self.agent.interact(
                        prompts=test_gen_batch,
                        mode='val',
                        global_steps=self.global_steps,
                    )
            
            test_sequence_score = test_gen_batch_output.batch['token_level_scores'].sum(-1) # (bs,)
            test_active_num_list = test_gen_batch_output.meta_info["active_num_list"]
            test_scores.append(test_sequence_score)
            test_all_active_num_lists.append(test_active_num_list)
        
        test_scores = torch.cat(test_scores) # (total_len_of_val_dataloader,)
        test_finished_ratio_each_turns = collect_active_num_list(test_all_active_num_lists)

        val_metrics = {
            "val/test_score": torch.mean(test_scores.float()).detach().item(),
            "distribution/val_finished_ratio_each_turns": test_finished_ratio_each_turns,
        }
        return val_metrics


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        # load checkpoint before doing anything
        self._load_checkpoint()
        self.global_steps = 1 # start from step 1

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        for epoch in range(self.config.trainer.total_epochs):
            for train_data in self.train_dataloader:
                print(f"Epoch {epoch}, Step {self.global_steps}")
                is_last_step = self.global_steps >= self.total_training_steps
                metrics = {}
                timing_raw = {}
                
                with _timer("step", timing_raw):
                    # ------------- rollout -------------
                    batch, gen_batch = self._prepare_data(train_data, split='train')
                    # agent set to train for ini train env
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.agent.interact(
                            prompts=gen_batch,
                            mode='train',
                            global_steps=self.global_steps,
                        )
                    
                    # repeat to align with repeated responses in rollout
                    batch = batch.union(gen_batch_output)
                    active_num_list = torch.tensor(gen_batch_output.meta_info["active_num_list"])
                    metrics.update({
                        "distribution/sample_level_scores": batch.batch["sample_level_scores"],
                        "distribution/finished_ratio_each_turns": 1 - active_num_list / active_num_list[0],
                    })

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ------------- forward -------------
                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        with torch.no_grad():
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl_in_reward,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                    
                    # ------------- backward -------------
                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                                
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # ------------- validate -------------
                    if (
                        self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    # ------------- save -------------
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                config = self.config
                n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
                # Implement actual tflpo and theoretical tflpo
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # TODO: make a canonical logger that supports various backend
                self.logger.log(data=metrics, step=self.global_steps)
                pprint(f"Train metrics at step {self.global_steps}: {metrics}")
                if is_last_step:
                    return

                self.global_steps += 1
