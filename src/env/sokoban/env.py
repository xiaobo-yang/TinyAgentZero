from typing import Dict, List
import numpy as np
from verl import DataProto

from .env_single import SokobanEnvSingle


class SokobanEnv:

    paraphrase = False 
    # if paraphrase is True, the env will return reflection cot on response_str, 
    # then these cot will be used as new thinking cot in agent

    def __init__(self, n_envs, ini_envs_from_data=False, max_steps=50, dim_room=(5,5), num_boxes=1, search_depth=30, seeds=None):
        """
        Envs may come from dataset, or generate from scratch when ini_envs_from_data is False.
        """
        self.max_steps = max_steps
        self.n_envs = n_envs
        self.rewards = {idx: [] for idx in range(n_envs)}
        self.trajs = {
            'trajectory': {idx: [] for idx in range(n_envs)},
            }
        self.ini_envs_from_data = ini_envs_from_data
        if not ini_envs_from_data:
            self.envs = [
                SokobanEnvSingle(
                    dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps, search_depth=search_depth, paraphrase=self.paraphrase
                ) for _ in range(n_envs)
            ]
            if seeds is not None:
                for env, seed in zip(self.envs, seeds):
                    env.reset(seed=seed, mode='tiny_rgb_array')

        else:
            self.envs = None

    def run(self, responses_str: List[str], batch: DataProto, chat_template=None):  # TODO: change to parallel
        """
        receive model response, run the environment
        """
        if self.ini_envs_from_data and self.envs is None:
            # load envs initial states from dataset
            assert len(batch) == self.n_envs, f"Number of environments ({len(batch)}) does not match the number of environments ({self.n_envs})"
            self.envs = []
            for data_item in batch:
                dim_room = data_item.non_tensor_batch['extra_info']['env_info']['dim_room']
                num_boxes = data_item.non_tensor_batch['extra_info']['env_info']['num_boxes']
                search_depth = data_item.non_tensor_batch['extra_info']['env_info']['search_depth']
                env = SokobanEnvSingle(
                    dim_room=dim_room, num_boxes=num_boxes, max_steps=self.max_steps, search_depth=search_depth
                )
                seed = data_item.non_tensor_batch['extra_info']['env_info']['seed']
                env.reset(seed=seed, mode='tiny_rgb_array')
                self.envs.append(env) 

        # update paraphase mode               
        for env in self.envs:
            env.paraphrase = self.paraphrase
        
        feedbacks, dones, step_rewards = [], [], []
        for idx, (env, response, data_item) in enumerate(zip(self.envs, responses_str, batch)):
            feedback, done = env.run(response, data_item, chat_template)
            step_reward = env.reward[-1]
            feedbacks.append(feedback)
            dones.append(done)
            step_rewards.append(step_reward)

            self.rewards[idx].append(step_reward)
            self.trajs['trajectory'][idx].append({
                'current_step': len(self.rewards[idx]),
                'response': response,
                'done': done,
                'step_reward': step_reward,
                'feedback': feedback,
            })

        # debug
        import random
        ridx = random.randint(0, len(step_rewards)-1)
        print(f"--------------------------------")
        print(f"Done: {dones[ridx]}, Step Reward: {step_rewards[ridx]}")
        print(f"Response: {responses_str[ridx]}")
        print(f"Feedback: {feedbacks[ridx]}")

        return responses_str, feedbacks, dones, step_rewards, self.trajs


