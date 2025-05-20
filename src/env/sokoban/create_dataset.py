import os
from datasets import Dataset
import argparse

from .env_single import SokobanEnvSingle


INSTRUCTION_TEMPLATE = """**Role**: Expert Sokoban puzzle-solving assistant. Analyze game states and provide optimal moves.

### Game Fundamentals
**Objective**: Push all boxes (X) onto target spots (O).

#### Symbols and Their Meaning
- **Walls (`#`)**: These block movement. You can't move through or push anything into walls.
- **Floor (`_`)**: Open spaces where you can walk and move boxes.
- **Target Spots (`O`)**: The spots where boxes need to go.
- **Boxes (`X`)**: These are what you need to push onto the targets.
- **Player (`P`)**: That's you! You'll move around the grid to push boxes.
- **Box on Target Spot (`√`)**: A box successfully placed on a target spot.
- **Player on Target Spot (`S`)**: You standing on a target spot.

### Core Rules
1. **Movement**
   - Each action is a move in one of the four cardinal directions: Up, Down, Left, Right
   - The movement can push the box if the box is in the same direction with the player
   - A valid Up move increases the vertical coordinate value by 1.
   - A valid Down move decreases the vertical coordinate value by 1.
   - A valid Left move decreases the horizontal coordinate value by 1.
   - A valid Right move increases the horizontal coordinate value by 1.
   - No diagonal/multi-box pushes
   - Cannot pull boxes
   - Walls block movement

2. **Deadlock Prevention**
   - Block moves that create:
   * Boxes in corner walls
   * Boxes in tunnel with no exit
   * Unmovable box formations

3. **Optimization**
   - Minimize total moves
   - Sequence box placements strategically

### Output Requirements
1. Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format.
2. In the <think> tag, retell each row of the observation and repeat the current layout row by row. Identify the current coordinate positions of targets, boxes, walls, and open spaces. Then, try each possible move and evaluate the potential result. Verify the evaluation based on the rules to ensure the layout is valid. Lastly select the best move based on the evaluation.
3. In the <answer> tag, output the action you want to take and the layout after your action. One move per response and valid moves are Up/Down/Left/Right.
### Reward System
| Action                | Value  |
|-----------------------|--------|
| Valid move            | -0.1   |
| Box on target         | +1.0   |
| Full solution         | +10.0  |
| Invalid move attempt  | -1.0   |

[Initial Observations]:
{observation}

Please decide the next action:
"""

templates = {
    'base': '\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format.',
}

if __name__ == "__main__":
# Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    parser.add_argument("--algo", type=str, default="bfs", choices=["bfs"], help="Algorithm to use (default: 'bfs').")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation.")
    parser.add_argument("--output", type=str, default="dataset/sokoban", help="Output file to save the trajectories.")
    parser.add_argument("--train_size", type=int, default=10000, help="Number of trajectories to generate.")
    parser.add_argument("--test_size", type=int, default=100, help="Number of trajectories to generate.")
    parser.add_argument("--bfs_max_nodes", type=int, default=1000, help="Maximum number of nodes to use for BFS.") # not using this now. This will usually give the best traj. To compare with SFT, we will try this later.
    parser.add_argument("--prefix", type=str, default='base', choices=['base']) # NOTE: Verl dataset will add template to the prompt, so we don't need to add template here.
    parser.add_argument("--dim_x", type=int, default=5, help="Dimension of the room.")
    parser.add_argument("--dim_y", type=int, default=5, help="Dimension of the room.")
    parser.add_argument("--num_boxes", type=int, default=1, help="Number of boxes in the room.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps.")
    parser.add_argument("--search_depth", type=int, default=30, help="Search depth for BFS.")
    args = parser.parse_args()
    
    assert args.algo == "bfs", "Unsupported algorithm: {args.algo}"
    data_source = 'sokoban'
    
    dim_x, dim_y, num_boxes, max_steps, search_depth = args.dim_x, args.dim_y, args.num_boxes, args.max_steps, args.search_depth

    seeds = range(args.seed, args.seed + args.train_size + args.test_size)
    instructions = []
    env_infos = []
    for seed in seeds:
        env = SokobanEnvSingle(
            dim_room=(dim_x, dim_y),
            num_boxes=num_boxes,
            max_steps=max_steps,
            search_depth=search_depth
        )
        observation = env.reset(seed=seed, mode='tiny_rgb_array')
        instruction = INSTRUCTION_TEMPLATE.format(observation=observation)
        instructions.append(instruction)
        env_info = {
            "dim_room": (dim_x, dim_y),
            "num_boxes": num_boxes,
            "max_steps": max_steps,
            "search_depth": search_depth,
            "seed": seed,
        }
        env_infos.append(env_info)

    
    def _create_instance(idx, instruction, env_info, split):
        prompt_formatted = templates[args.prefix].format(prompt=instruction)

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "extra_info": {"split": split, "index": idx, "env_info": env_info}
        }
    train_dataset = Dataset.from_list([_create_instance(args.seed + i, instructions[i], env_infos[i], 'train') for i in range(args.train_size)])
    test_dataset = Dataset.from_list([_create_instance(args.seed + i, instructions[i], env_infos[i], 'test') for i in range(args.train_size, args.train_size + args.test_size)])


    def make_map_fn(split):
        def process_fn(example, idx):
            return example
        return process_fn

    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))