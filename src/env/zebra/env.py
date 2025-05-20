from typing import List
import numpy as np

from .env_single import ZebraEnvSingle



class ZebraEnv:

    def __init__(self, n_envs):
        """
        Initialize the environment for Zebra Puzzle
        """
        self.n_envs = n_envs
        self.envs = [ZebraEnvSingle() for _ in range(n_envs)]
        self.rewards = {idx: [] for idx in range(n_envs)}
        self.trajs = {
            'golden_answer': {idx: None for idx in range(self.n_envs)},
            'trajectory': {idx: [] for idx in range(self.n_envs)},
            }

    def run(self, responses_str: List[str], batch, chat_template=None):  # TODO: change to parallel
        """
        receive model response, run the environment
        """
        feedbacks, dones, step_rewards = [], [], []
        for idx, (env, response, data_item) in enumerate(zip(self.envs, responses_str, batch)):
            if self.trajs['golden_answer'][idx] is None:
                golden_answer = data_item.non_tensor_batch['extra_info']["puzzle_data"]['solution']
                filtered_answer = {} # avoid json.dumps error for np.array in dict
                for k, v in golden_answer.items():
                    if v is not None and isinstance(v, np.ndarray):
                        filtered_answer[k] = v.tolist()
                self.trajs['golden_answer'][idx] = filtered_answer

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