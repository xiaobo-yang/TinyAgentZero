from .sokoban.env import SokobanEnv
from .search.env import SearchEnv
from .zebra.env import ZebraEnv
from .math.env import MathEnv
from .code.env import CodeEnv

def get_env(config: dict, n_envs, split, paraphrase=False):
    """
    Prepare environments for agent training.
    """
    env_name = config.env.name
    if env_name == "sokoban":
        env = SokobanEnv(
            n_envs=n_envs,
            ini_envs_from_data=True,
        )
    elif env_name == "search":
        env = SearchEnv(
            n_envs=n_envs,
            search_url=config.env.search_url, 
            topk=config.env.search_topk, 
            is_validation=False if split == 'train' else True
        )
    elif env_name == "zebra":
        env = ZebraEnv(
            n_envs=n_envs,
        )
    elif env_name == "math":
        env = MathEnv(
            n_envs=n_envs,
            is_validation=False if split == 'train' else True
        )
    elif env_name == "code":
        env = CodeEnv(
            n_envs=n_envs,
        )
    else:
        raise ValueError(f"Environment {config.env.name} not supported")
    
    # if use paraphrase, the env will return reflection cot on response_str, not direct answer
    if paraphrase:
        env.paraphrase = True
    else:
        env.paraphrase = False

    return env