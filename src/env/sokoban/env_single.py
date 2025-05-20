import re
from typing import Any, Dict
import numpy as np
import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
from verl.protocol import DataProtoItem

from .room_utils import generate_room
from .utils import set_seed, NoLoggerWarnings


class SokobanEnvSingle(GymSokobanEnv):

    GRID_LOOKUP = {
        0: " # \t",  # wall
        1: " _ \t",  # floor
        2: " O \t",  # target
        3: " √ \t",  # box on target
        4: " X \t",  # box
        5: " P \t",  # player
        6: " S \t",  # player on target
        # Use tab separator to separate columns and \n\n to separate rows.
    }
    ACTION_LOOKUP = {
        0: "None",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1

    def __init__(self, **kwargs):
        self.reward = []
        self._actions = [] # list of all actions (including all responses from LLM)
        self._actions_valid = [] # list of actions that are in the correct format
        self._actions_effective = [] # list of actions that are effective (actual moving in env)
        self.cur_seq = []
        self.action_sequence = []
        self.search_depth = kwargs.pop('search_depth', 300)
        GymSokobanEnv.__init__(
            self,
            dim_room=kwargs.pop('dim_room', (6, 6)), 
            max_steps=kwargs.pop('max_steps', 100),
            num_boxes=kwargs.pop('num_boxes', 3),
            **kwargs
        )
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self._valid_actions = []
        self.paraphrase = kwargs.pop('paraphrase', False)

    def run(self, response: str, data_item: DataProtoItem, chat_template=None):
        """
        receive model response, run the environment. chat_template is used for feedback
        """
        feedback = ""
        

        if self.finished():
            done = True
            self.reward.append(0.0)
        else:
            if chat_template is not None:
                end_pattern = f"{chat_template['end']}"
                # add end pattern to response if not exist
                if end_pattern not in response:
                    feedback = f"{chat_template['end']}"

            action = self.extract_action(response)
            observation, reward, done, extra_info = self.execute(action)
            feedback_info = self.postprocess(action, observation, reward, done)

            if chat_template is not None:
                feedback += (
                    f"\n{chat_template['start']}{chat_template['user']}\n{feedback_info}{chat_template['end']}\n"
                    f"{chat_template['start']}{chat_template['assistant']}\n{chat_template['think_start']}"
                )
            else:
                feedback += feedback_info

            self.update_env_variables(
                response=response, 
                action=action, 
                action_is_effective=extra_info.get("action_is_effective", False), 
                reward=reward, 
            )

        return feedback, done
    
    def extract_action(self, response: str):
        """
        Extract action from query.
        - 0: Still (Invalid Action)
        - 1: Up
        - 2: Down
        - 3: Left
        - 4: Right
        """
        # extract from <answer></answer>
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            query = match.group(1).strip()
        else:
            return self.INVALID_ACTION

        DIRECTION_MAP = {"Up": 1, "Down": 2, "Left": 3, "Right": 4}
        # TODO: originally, we parse either number (key of direction_map) or direction (value of direction_map).
        # here we remove numbers and preserve directions only, but regex has not been removed. please remove them later.
        pattern = r'^\s*(([1-4])\s*\((up|down|left|right)\)|(up|down|left|right)|([1-4]))\s*$'
        match = re.fullmatch(pattern, query.strip(), flags=re.IGNORECASE | re.X)
        
        if not match:
            return self.INVALID_ACTION
        
        if match.group(2):   
            return int(match.group(2))
        elif match.group(4): 
            return DIRECTION_MAP[match.group(4).capitalize()]
        elif match.group(5): 
            return int(match.group(5))
        
        return self.INVALID_ACTION
    
    def execute(self, action: int):
        """
        - Step the environment with the given action.
        - Check if the action is effective (whether player moves in the env).

        Output:
            observation, env_reward, done, extra_info
        """
        assert not self.success()

        if action == self.INVALID_ACTION:
            return self.render(), 0, False, {"action_is_effective": False}
        else:
            prev_player_position = self.player_position
            _, reward, done, _ = self.step(action, observation_mode='tiny_rgb_array')
            obs = self.render()
            return obs, reward, done, {"action_is_effective": not np.array_equal(prev_player_position, self.player_position)}

    def reset(self, mode='tiny_rgb_array', seed=None):
        
        self._reset_tracking_variables()
        with NoLoggerWarnings():
            try:
                with set_seed(seed):
                    self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                        dim=self.dim_room,
                        num_steps=self.num_gen_steps,
                        num_boxes=self.num_boxes,
                        search_depth=self.search_depth
                    )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
                return self.reset(mode, next_seed)
            
            self.player_position = np.argwhere(self.room_state == 5)[0]
            self.num_env_steps = self.reward_last = self.boxes_on_target = 0
            return self.render(mode)

    def success(self):
        return self.boxes_on_target == self.num_boxes

    def finished(self):
        return self.success()  
    
    def postprocess(self, action: int, observation, reward, done):
        if action == self.INVALID_ACTION:
            feedback_info = f"Action is invalid. You stay in the same position. One move per response and valid moves are Up/Down/Left/Right. The observation is: \n{observation}\nreward: {reward}\ndone: {done}\n"
        else:
            feedback_info = f"After you take this action, the observation is: \n{observation}\nreward: {reward}\ndone: {done}\n"

        if self.paraphrase:
            feedback_info = (
                f'After you take action {action}, the environment feedback is: "{feedback_info}"\n'
                f'Reflect on any mistakes or shortcomings in your action. Consider how to improve your approach to complete this task successfully.'
            )

        return feedback_info

    
    def update_env_variables(
            self, 
            response: str,
            action: Any, 
            action_is_effective: bool,
            reward: float,
        ):
        """
        All of _actions, _actions_valid, _actions_effective are lists of the same length
            - None is used for _actions_valid and _actions_effective if the action is invalid or ineffective
        """
        self._actions.append(response)
        action_is_valid = action != self.INVALID_ACTION
        
        if action_is_valid:
            self._actions_valid.append(action)
        else:
            self._actions_valid.append(None)
        
        if action_is_effective:
            self._actions_effective.append(action)
        else:
            self._actions_effective.append(None)
        
        self.reward.append(reward if action_is_valid else (reward + self.PENALTY_FOR_INVALID))

    def _reset_tracking_variables(self):
        self.reward = []
        self._actions = []
        self._actions_valid = []
        self._actions_effective = []

    def render(self, mode='tiny_rgb_array'):
        assert mode in ['tiny_rgb_array', 'list', 'state', 'rgb_array']

        if mode == 'rgb_array':
            img = self.get_image(mode, scale=1) # numpy array
            return img


        if mode == 'state':
            return np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        
        room_state = self.render(mode='state').tolist()

        if mode == 'list':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in room_state]
        
        if mode == 'tiny_rgb_array':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            return "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
