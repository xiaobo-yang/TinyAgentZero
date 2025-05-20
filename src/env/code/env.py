from typing import List
from concurrent.futures import ThreadPoolExecutor

from .reward_function import rllm_reward_fn_code

class CodeEnv:

    def __init__(self, n_envs, *args, **kwargs):
        """
        Initialize the environment for Zebra Puzzle
        """
        self.n_envs = n_envs
        self._success = {i: False for i in range(n_envs)}
        self.reward = {idx: [] for idx in range(self.n_envs)}
        self.trajs = {
            'question': {idx: None for idx in range(self.n_envs)},
            'trajectory': {idx: [] for idx in range(self.n_envs)},
            }

    def run(self, responses_str: List[str], batch, chat_template=None):
        """
        receive model response, run the environment
        """
        def process_item(args):
            finished, sequences_str, data_source, ground_truth = args
            if not finished:
                score = rllm_reward_fn_code(data_source=data_source, llm_solution=sequences_str, ground_truth=ground_truth)
            else:
                score = 0.0
            return score

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=48) as executor:
            args = [
                (finished, sequences_str, extra_info['data_source'], extra_info['ground_truth']) for finished, sequences_str, extra_info in zip(
                    list(self._success.values()), responses_str, batch.non_tensor_batch['extra_info']
                )
            ]
            step_rewards = list(executor.map(process_item, args))
        
        feedbacks, dones = [], []
        for i, (r, extra_info) in enumerate(zip(step_rewards, batch.non_tensor_batch['extra_info'])):
            if self._success[i]:
                feedback = ""
                done = True
            elif not self._success[i] and r >= 1.0:
                self._success[i] = True
                feedback = ""
                done = True
            else:
                observation = (
                    'Your answer is incorrect! You need to try another answer. '
                )
                if chat_template is not None:
                    observation = (
                        f"\n{chat_template['start']}{chat_template['user']}\n{observation}{chat_template['end']}\n"
                        f"{chat_template['start']}{chat_template['assistant']}"#\n{chat_template['think_start']}"  # TODO: add think in the deepcoder prompt
                    )
                    end_pattern = f"{chat_template['end']}"
                    if end_pattern not in responses_str[i]:
                        observation = f"{chat_template['end']}" + observation
                feedback = observation
                done = False
            
            feedbacks.append(feedback)
            dones.append(done)
            self.reward[i].append(r)
            self.trajs['trajectory'][i].append({
                'current_step': len(self.reward[i]),
                'done': done,
                'step_reward': r,
                'ground_truth': extra_info['ground_truth'],
                'filtered response': responses_str[i],
                'feedback': feedback,
            })

        # debug
        import random
        ridx = random.randint(0, len(self.reward)-1)
        print(f"--------------------------------")
        print(f"Filtered response: {responses_str[ridx]}")

        return responses_str, feedbacks, dones, step_rewards, self.trajs
