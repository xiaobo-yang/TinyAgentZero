from typing import List


class TemplateEnv:

    def __init__(self, max_turns, n_envs, *args, **kwargs):
        """
        Initialize the environment for Zebra Puzzle
        """
        self.n_envs = n_envs
        self.max_turns = max_turns
        self.trajs = {}

    def run(self, responses_str: List[str], batch, chat_template=None):
        """
        receive model response, run the environment
        """
        feedbacks, dones, step_rewards = [], [], []

        # Implement your environment interaction logic here

        return responses_str, feedbacks, dones, step_rewards, self.trajs
