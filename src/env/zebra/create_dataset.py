"""
Create dataset for Zebra puzzle environment.
"""

import os
import time
import random
import argparse
from typing import List, Dict, Any, Tuple
from datasets import Dataset
import tqdm

from zebra_generator import generate_puzzle, generate_puzzle_data

templates = {
    'base': '\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format.',
}

intro = """You are solving a Zebra puzzle (also known as Einstein's Puzzle).

## Description
In this puzzle, you need to match attributes to objects based on given clues.
Each attribute (like houses, colors, drinks, etc.) must be matched to exactly one item in each category.
You'll have chances to improve your answer. After each response, you'll receive feedback. Use it to enhance your answer.

## How to Solve
1. Use logical deduction to determine which attributes go together
2. Create a table or chart to track possibilities
3. Use clues to eliminate impossible combinations
4. Look for both direct and indirect relationships

## Format Your Answer
Provide your solution in this format:
<answer>
| House | Attribute 1 | Attribute 2 | ... |
| --- | --- | --- | --- |
| 1 | [attribute1] | [attribute2] | ... |
| 2 | [attribute1] | [attribute2] | ... |
| ... | ... | ... | ... |
</answer>

## Puzzle:
{puzzle_description}

Clues:
{clues}

Solve the puzzle and determine the correct arrangement:
"""

def format_puzzle_description(table: List[List[str]], n_attributes: int, m_objects: int) -> str:
    """Format puzzle description from table with randomized attribute order."""
    description = f"There are {m_objects} houses, numbered from 1 to {m_objects}. Each house has a different owner, and each person has {n_attributes} characteristics. These possible characteristics are as follows:\n"
    
    # Create a copy of the table and shuffle it to randomize attribute order
    shuffled_table = table.copy()
    random.shuffle(shuffled_table)
    
    for row in shuffled_table:
        # For each attribute type, also randomize the order of possible values
        values = row[1:].copy()
        random.shuffle(values)
        description += f"- {row[0]}: {', '.join(values)}\n"
    
    return description

def format_clues(premises: List[str]) -> str:
    """Format clues with index numbers."""
    result = ""
    for i, premise in enumerate(premises, 1):
        result += f"{i}. {premise}\n"
    return result

def create_puzzle_instance(n_attributes: int, m_objects: int, level: int, seed: int) -> Tuple[Dict[str, Any], str]:
    """
    Create a single puzzle instance.
    
    Args:
        n_attributes: Number of attributes
        m_objects: Number of objects per attribute
        level: Difficulty level of the puzzle
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (puzzle_data, formatted_prompt)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Generate puzzle table and premises
    t1 = time.monotonic()
    table = []
    premises = []
    
    try:
        # Call the generator with the specific seed
        old_state = random.getstate()
        random.seed(seed)
        
        # Generate puzzle data
        table = generate_puzzle_data(n_attributes, m_objects, level, minimal_conditions=True)
        # Get the premises for the puzzle
        premises = generate_puzzle(table, level=level, minimal_conditions=True, max_seconds_for_minimizing=10)
        
        # Restore random state
        random.setstate(old_state)
    except Exception as e:
        print(f"Error generating puzzle with seed {seed}: {e}")
        return None, None
    
    t2 = time.monotonic()
    
    # Format puzzle description and clues
    puzzle_description = format_puzzle_description(table, n_attributes, m_objects)
    clues_text = format_clues(premises)
    
    # Create prompt with formatted puzzle
    prompt = intro.format(
        puzzle_description=puzzle_description,
        clues=clues_text
    )
    
    # Create solution table where each column represents a house
    attributes = [row[0] for row in table]
    solution = {}
    
    # Transpose the table to have house-centric data
    for i, attr in enumerate(attributes):
        solution[attr] = table[i][1:]  # Skip the attribute name
    
    # Create the puzzle data dictionary
    puzzle_data = {
        "attributes": attributes,
        "objects": {row[0]: row[1:] for row in table},
        "rules": premises,
        "solution": solution,
        "level": level
    }
    
    return puzzle_data, prompt

def main():
    parser = argparse.ArgumentParser(description="Generate dataset for Zebra puzzle environment.")
    parser.add_argument("--output", type=str, default="dataset/zebra", help="Output directory.")
    parser.add_argument("--train_size", type=int, default=10000, help="Number of training instances.")
    parser.add_argument("--test_size", type=int, default=100, help="Number of test instances.")
    parser.add_argument("--n_attributes", type=int, default=3, help="Number of attributes (default: 3).")
    parser.add_argument("--m_objects", type=int, default=4, help="Number of objects per attribute (default: 4).")
    parser.add_argument("--min_level", type=int, default=1, help="Minimum difficulty level (default: 1).")
    parser.add_argument("--max_level", type=int, default=10, help="Maximum difficulty level (default: 10).")
    parser.add_argument("--test_level_increment", type=int, default=1, help="How much to increase difficulty for test set (default: 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--prefix", type=str, default='base', choices=['base'])
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    # Set random seed
    random.seed(args.seed)
    
    # Generate train and test datasets
    train_instances = []
    test_instances = []
    
    print(f"Generating {args.train_size} training instances with mixed levels ({args.min_level}-{args.max_level})...")
    for i in tqdm.tqdm(range(args.train_size), desc="Training instances"):
        seed_value = args.seed + i
        # Select a random level for each training instance
        level = random.randint(args.min_level, args.max_level)
        puzzle_data, prompt = create_puzzle_instance(
            args.n_attributes, args.m_objects, level, seed_value
        )
        
        if puzzle_data is not None:
            train_instances.append((seed_value, puzzle_data, prompt))
    
    print(f"Generating {args.test_size} test instances...")
    for i in tqdm.tqdm(range(args.test_size), desc="Test instances"):
        seed_value = args.seed + args.train_size + i
        # For test set, also use random levels but with a slight increase in difficulty
        base_level = random.randint(args.min_level, args.max_level)
        test_level = min(base_level + args.test_level_increment, 20)  # Max level is 20
        puzzle_data, prompt = create_puzzle_instance(
            args.n_attributes, args.m_objects, test_level, seed_value
        )
        
        if puzzle_data is not None:
            test_instances.append((seed_value, puzzle_data, prompt))
    
    def _create_instance(idx, puzzle_data, prompt):
        """Create a dataset instance in the required format."""
        prompt_formatted = templates[args.prefix].format(prompt=prompt)
        return {
            "data_source": "zebra",
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "deduction",
            "extra_info": {"split": "train", "index": idx, "puzzle_data": puzzle_data}
        }
    
    # Create datasets
    train_dataset = Dataset.from_list([
        _create_instance(seed, puzzle_data, prompt) 
        for seed, puzzle_data, prompt in train_instances
    ])
    
    test_dataset = Dataset.from_list([
        _create_instance(seed, puzzle_data, prompt) 
        for seed, puzzle_data, prompt in test_instances
    ])
    
    def make_map_fn(split):
        def process_fn(example, idx):
            example["extra_info"]["split"] = split
            return example
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    # Save datasets
    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))
    
    print(f"Successfully created {len(train_dataset)} training and {len(test_dataset)} test instances.")
    print(f"Datasets saved to {args.output}")

if __name__ == "__main__":
    main()