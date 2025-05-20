from .reward_function import compute_score



class ZebraEnvSingle:
    """
    Zebra puzzle (also known as Einstein's Puzzle) environment.

    ## Description
    In this puzzle, the agent must solve a constraint satisfaction problem
    where multiple attributes need to be matched correctly based on given clues.
    Each attribute (like houses, colors, drinks, etc.) must be matched to exactly one
    item in each category.

    ## Action Space
    For Zebra puzzle, the action space is text input containing the solution.

    ## Rewards
    - Format reward (if correct format, but wrong answer): 0.2
    - Correct answer: 1.0

    ## Observation
    The puzzle description and clues.

    NOTE only one step
    """


    def __init__(self):
        """
        Initialize the environment for Zebra Puzzle

        Args:
            puzzle_data: Dict containing the puzzle description, attributes, and solution
        """
        self._success = False
        self.n_turn = 0
        self.reward = []

    def run(self, response: str, data_item, chat_template=None):
        """
        receive model response, run the environment
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
            
            query = self.extract_action(response)
            puzzle_data = data_item.non_tensor_batch['extra_info']["puzzle_data"]
            observation, done = self.execute(query, puzzle_data)
            
            if chat_template is not None:
                feedback += (
                    f"\n{chat_template['start']}{chat_template['user']}\n{observation}{chat_template['end']}\n"
                    f"{chat_template['start']}{chat_template['assistant']}\n{chat_template['think_start']}"
                )
            else:
                feedback += observation

        
        self.n_turn += 1

        return feedback, done

    def extract_action(self, response: str) -> str:
        """
        Extract action from text, text-based input is valid if not empty
        """
        return response.strip()
    
    def execute(self, query: str, puzzle_data):
        """
        Output:
            observation, done

        Args:
            puzzle_data: Dict containing the puzzle description, attributes, and solution
        """
        assert not self.finished()

        # Calculate reward based on solution correctness
        reward, feedback = compute_score(query, puzzle_data)
        self.reward.append(reward)

        
        self._success = reward == 1.0
        if self._success:
            done = True
        else:
            done = False

        attributes = puzzle_data.get("attributes", [])
        objects = puzzle_data.get("objects", {})
        rules = puzzle_data.get("rules", [])
        restate_attributes = ""
        for attr in attributes:
            restate_attributes += f"{attr.capitalize()}: {', '.join(objects.get(attr, []))}\n"
        restate_clues = ""
        for i, rule in enumerate(rules):
            restate_clues += f"{i+1}. {rule}\n"

        observation = f"Original attributes:\n{restate_attributes}\nOriginal clues:\n{restate_clues}\n{feedback}"

        return observation, done

    def reset(self):
        """Reset the environment"""
        self._success = False
        self.n_turn = 0
        self.reward = []


    def success(self):
        return self._success

    def finished(self):
        return self.success()
    


