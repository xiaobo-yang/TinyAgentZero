import logging
import os
import torch
from omegaconf import open_dict, DictConfig
from verl import DataProto
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu
from verl.utils.import_utils import import_external_libs
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class PatchedActorRolloutRefWorker(ActorRolloutRefWorker):

    def __init__(self, config: DictConfig, role: str):
        old_role = role
        if old_role == "ref_rollout":
            role = "ref"
        super().__init__(config, role)
        if old_role != "ref_rollout":
            self._is_ref_rollout = False
            return
        role = old_role

        # NOTE: monkey patch for ref module with rollout
        self.role = role
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref", "ref_rollout"] # NOTE: add rollout for ref
        self._is_ref = self.role in ["ref", "actor_rollout_ref", "ref_rollout"]
        self._is_ref_rollout = self.role in ["ref_rollout"]

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} "
                "should be larger than 0 after normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be "
                    f"divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be "
                    f"larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if (self._is_rollout or self._is_ref_rollout) and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if (self._is_ref or self._is_ref_rollout) and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        from src.patch.actor_patch import update_policy_with_info_mask  # NOTE: monkey patch
        DataParallelPPOActor.update_policy = update_policy_with_info_mask
        print("Monkey patch DataParallelPPOActor.update_policy")

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout or self._is_ref_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            if self._is_ref_rollout:
                self.ref_module_fsdp, _, _, self.ref_model_config = self._build_model_optimizer(model_path=self.config.model.path,
                                                                    fsdp_config=self.config.ref.fsdp_config,
                                                                    optim_config=None,
                                                                    override_model_config=override_model_config,
                                                                    use_remove_padding=use_remove_padding,
                                                                    trust_remote_code=self.config.model.get(
                                                                        'trust_remote_code', False),
                                                                    use_liger=self.config.model.get('use_liger', False),
                                                                    role='ref')
                OmegaConf.set_struct(self.config.ref, True)
                with open_dict(self.config.ref):
                    self.config.ref.use_remove_padding = use_remove_padding
                self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
                
                # NOTE: actor_module is used for rollout sync
                self.actor_model_config = self.ref_model_config
                self.actor_module_fsdp = self.ref_module_fsdp
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module
            else:
                self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                    model_path=self.config.model.path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                    trust_remote_code=self.config.model.get('trust_remote_code', False),
                    use_liger=self.config.model.get('use_liger', False),
                    role='actor')

                # get the original unwrapped module
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

                if self._is_offload_optimizer:
                    offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                    log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                                actor_module=self.actor_module_fsdp,
                                                actor_optimizer=self.actor_optimizer)

        if self._is_rollout or self._is_ref_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(
                trust_remote_code=self.config.model.get('trust_remote_code', False))

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                                fsdp_config=self.config.ref.fsdp_config,
                                                                optim_config=None,
                                                                override_model_config=override_model_config,
                                                                use_remove_padding=use_remove_padding,
                                                                trust_remote_code=self.config.model.get(
                                                                    'trust_remote_code', False),
                                                                use_liger=self.config.model.get('use_liger', False),
                                                                role='ref')[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents)
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(torch.cuda.current_device())

        assert self._is_rollout or self._is_ref_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # clear kv cache
        torch.cuda.empty_cache()
        return output
        
class PatchedCriticWorker(CriticWorker):
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from transformers import set_seed  # NOTE: monkey patch
        set_seed(42)
        print("Monkey patch set_seed before critic init model")

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_contents=self.config.checkpoint.contents)









