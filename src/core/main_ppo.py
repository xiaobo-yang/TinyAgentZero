import os
import importlib.util
from pathlib import Path
import ray
import hydra
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

from src.core.agent_trainer import RayAgentTrainer


verl_spec = importlib.util.find_spec("verl")
verl_path = Path(os.path.dirname(verl_spec.origin)).parent
config_path = os.path.join(verl_path, "verl/trainer/config") # default config in verl package
@hydra.main(config_path=config_path, config_name="ppo_trainer", version_base=None)
def main(config):
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            address="localhost:6379",
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true", 
                    "NCCL_DEBUG": "WARN", 
                    "VLLM_LOGGING_LEVEL": "WARN",
                    'RAY_DEBUG_POST_MORTEM': '1',  # use ray debugger
                    }
            }
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local

        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from src.patch.workers_patch import PatchedActorRolloutRefWorker, PatchedCriticWorker

            from verl.single_controller.ray import RayWorkerGroup

            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not implemented")

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(PatchedActorRolloutRefWorker),
            Role.Critic: ray.remote(PatchedCriticWorker),
            Role.RefPolicy: ray.remote(PatchedActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        ref_rollout_pool_id = "ref_rollout_pool"  # NOTE: need another pool for ref rollout, vllm sleep mode can be used only once on one process
        if config.algorithm.paraphrase.enable:
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node // 2] * config.trainer.nnodes,
                ref_rollout_pool_id: [config.trainer.n_gpus_per_node // 2] * config.trainer.nnodes,
            }
            mapping = {
                Role.ActorRollout: global_pool_id,
                Role.Critic: global_pool_id,
                Role.RefPolicy: ref_rollout_pool_id,
            }
        else:
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
            mapping = {
                Role.ActorRollout: global_pool_id,
                Role.Critic: global_pool_id,
                Role.RefPolicy: global_pool_id,
            }

        if config.reward_model.enable:
            raise NotImplementedError("Reward model is not implemented")


        # backend
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # trainer
        trainer = RayAgentTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
        )
        trainer.fit()


if __name__ == "__main__":
    main()
