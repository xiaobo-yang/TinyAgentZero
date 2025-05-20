# Quick Start

```shell
conda create -n agent python=3.10 -y
conda activate agent

# Install dependencies
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -r requirements.txt

# Add verl as a submodule
git submodule add https://github.com/volcengine/verl.git verl
git submodule update --init --recursive --remote
cd verl
pip3 install -e .
cd ..
```

# Supported Environments

- Sokoban
- Search
- Zebra Puzzle
- Math
- Code

# Features

- Clear support for custom environments
- Easy integration with verl via submodule
- Rephrased environment feedback
- Combined SFT and RL loss
- Monte Carlo Tree Search (MCTS)

# How to Add a New Environment

1. Create a new folder under `src/env/` (e.g., `myenv/`) and implement `env.py` following the examples in other environments.
2. Your environment class should implement at least:
   - `run(responses_str, batch, chat_template)`: main agent-environment interaction.
   - `get_reward_allocation(reward_tensors)`: (optional) custom reward allocation.
3. Register your environment in `src/env/__init__.py` and add a branch in the `get_env` function.
4. Specify your environment in training scripts with `+env.name=your_env`.

# Dataset Generation

Each environment provides a `create_dataset.py` script for generating training and test datasets. For example:
```shell
cd src/env/sokoban
python create_dataset.py --output dataset/sokoban --train_size 10000 --test_size 100
```
Other environments (math, zebra, code) are similar. See each script for available arguments.

# Training & Validation

For example, to train on the math environment (see `train_math_ppo.sh` for details):
```shell
bash train_math_ppo.sh
```
Or run the main program directly:
```shell
python -m src.core.main_ppo +env.name=math ...
```
See the training scripts for all configurable parameters.


# References
- TinyZero: https://github.com/Jiayi-Pan/TinyZero
- RAGEN: https://github.com/RAGEN-AI/RAGEN/
- Search-R1: https://github.com/PeterGriffinJin/Search-R1
- rllm: https://github.com/agentica-project/rllm
