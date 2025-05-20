"""
Reward function for zebra puzzle.
"""

import re
from typing import Dict, Any, List, Tuple

def parse_markdown_solution(solution_text, attributes):
    """
    Parse a markdown table solution into a list of lists ordered by attributes.

    Args:
        solution_text (str): The markdown table solution
        attributes (list): List of attribute names in the desired order

    Returns:
        list: List of lists, each inner list contains values for one attribute
    """
    # Clean the input solution text
    solution_text = solution_text.strip()

    # Extract answer content if surrounded by <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_text, re.DOTALL)
    
    if answer_match:
        solution_text = answer_match.group(1).strip()
    else:
        return None
    
    if len(solution_text) == 0:
        return None

    # Split into lines
    lines = solution_text.split("\n")

    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]

    # Extract header row and data rows
    if "|" in lines[0]:
        # This is a markdown table with vertical bars
        # Remove outer pipes and split by remaining pipes
        header = [cell.strip() for cell in lines[0].strip("|").split("|")]

        # Skip the separator line (contains only dashes and pipes)
        data_start_idx = 2 if len(lines) > 1 and all(c in "|-" for c in lines[1]) else 1

        # Extract data rows
        data_rows = []
        for line in lines[data_start_idx:]:
            if "|" in line:
                row = [cell.strip() for cell in line.strip("|").split("|")]
                data_rows.append(row)
    else:
        # Assume space-separated format
        header = lines[0].split()
        data_rows = [line.split() for line in lines[1:]]

    # If header doesn't match attributes, use positional mapping instead
    if len(header) > 1 and len(attributes) > 1:
        # Skip 'House' or position column if present
        start_idx = 1 if header[0].lower() in ['house', 'position', '#', 'no', 'no.'] else 0
        
        # Initialize result with empty lists for each attribute
        result = [[] for _ in range(len(attributes))]
        
        # Map columns by position, skipping the house/position column
        attr_indices = list(range(min(len(header) - start_idx, len(attributes))))
        
        # Extract values based on position
        for row in data_rows:
            for i, attr_idx in enumerate(attr_indices):
                col_idx = i + start_idx
                if col_idx < len(row):
                    # Include empty cells (just skip the "---" placeholder)
                    if row[col_idx] != "---":
                        result[attr_idx].append(row[col_idx])
                else:
                    # If the cell doesn't exist, add an empty string
                    result[attr_idx].append("")
        
        # Make sure all attribute lists have the same length (number of houses)
        max_length = max(len(attr_list) for attr_list in result) if result else 0
        for attr_list in result:
            while len(attr_list) < max_length:
                attr_list.append("")
        
        return result
    
    # Fall back to original attribute-based matching if needed
    # Create a mapping from header to column index
    header_indices = {header[i].lower(): i for i in range(len(header))}

    # Initialize result with empty lists for each attribute
    result = [[] for _ in range(len(attributes))]

    # Fill in the result based on the ordering in attributes
    for attr_idx, attr in enumerate(attributes):
        # Find the corresponding column in the table
        attr_lower = attr.lower()
        matching_cols = [col for col in header_indices.keys() if attr_lower in col]

        if matching_cols:
            col_idx = header_indices[matching_cols[0]]
            # Extract values for this attribute from all rows
            for row in data_rows:
                if col_idx < len(row):
                    # Include empty cells (just skip the "---" placeholder)
                    if row[col_idx] != "---":
                        result[attr_idx].append(row[col_idx])
                else:
                    # If the cell doesn't exist, add an empty string
                    result[attr_idx].append("")
    
    # Make sure all attribute lists have the same length (number of houses)
    max_length = max(len(attr_list) for attr_list in result) if result else 0
    for attr_list in result:
        while len(attr_list) < max_length:
            attr_list.append("")
    
    return result

def format_ground_truth_for_comparison(ground_truth: Dict[str, Any]) -> List[List[str]]:
    """
    Format the ground truth solution into a list of lists for comparison with parsed solution.
    
    Args:
        ground_truth: The ground truth data dictionary
    
    Returns:
        List of lists, each inner list contains values for one attribute
    """
    true_solution = ground_truth["solution"]
    attributes = ground_truth["attributes"]
    
    formatted_solution = []
    for attr in attributes:
        if attr in true_solution and true_solution[attr] is not None:
            # Convert to a normal list regardless of original type (array or list)
            attr_values = list(true_solution[attr])
            formatted_solution.append(attr_values)
        else:
            formatted_solution.append([])
    
    return formatted_solution

def validate_parsed_lists(parsed_lists: List[List[str]], ground_truth: Dict[str, Any]) -> Tuple[bool, float, str]:
    """
    Validate the parsed solution lists against the ground truth.
    
    Args:
        parsed_lists: The parsed solution as a list of lists, each inner list containing values for one attribute
        ground_truth: The ground truth data
    
    Returns:
        Tuple of (is_valid, score, feedback)
    """
    # Get ground truth solution and attributes
    true_solution = ground_truth["solution"]
    attributes = ground_truth["attributes"]
    objects_dict = ground_truth["objects"]
    
    # Format and print the ground truth solution for comparison (for debugging only)
    formatted_truth = format_ground_truth_for_comparison(ground_truth)
    # print(f"Ground truth solution: {formatted_truth}")
    # print(f"Parsed solution: {parsed_lists}")
    
    # 1. Check format
    # Check if the number of attributes matches
    if len(parsed_lists) != len(attributes):
        return False, 0.0, f"Format error: Expected {len(attributes)} attributes, got {len(parsed_lists)}"
    
    # Check if any attribute list is empty
    for i, attr_list in enumerate(parsed_lists):
        if not attr_list:
            return False, 0.0, f"Format error: Attribute list for {attributes[i]} is empty"
    
    # Expected number of houses
    expected_houses = len(objects_dict[attributes[0]])
    
    # Check if all attribute lists have the same length
    for i, attr_list in enumerate(parsed_lists):
        if len(attr_list) != expected_houses:
            return False, 0.0, f"Format error: Attribute {attributes[i]} has {len(attr_list)} values, expected {expected_houses}"
    
    # 2. Perform attribute matching to handle inexact attribute names
    attribute_mapping = {}  # Maps ground truth attribute indices to parsed attribute indices
    
    # Try to match each ground truth attribute to the closest parsed attribute
    for i, attr in enumerate(attributes):
        attr_lower = attr.lower()
        # Find the best match among parsed attributes
        best_match_idx = i  # Default to same index if no better match found
        best_match_score = 0
        
        for j, parsed_attr_list in enumerate(parsed_lists):
            # Calculate similarity based on values
            # Check overlap between expected values and parsed values
            expected_values_set = set(v.lower() for v in formatted_truth[i])
            parsed_values_set = set(v.lower() for v in parsed_attr_list)
            
            overlap = len(expected_values_set.intersection(parsed_values_set))
            if overlap > best_match_score:
                best_match_score = overlap
                best_match_idx = j
        
        attribute_mapping[i] = best_match_idx
    
    # 3. Compute score based on correct cells
    total_cells = expected_houses * len(attributes)
    correct_cells = 0
    cell_status = []  # Track which cells are correct
    values_in_solution = []  # Track which values are in solution but in wrong place
    
    for i, attr in enumerate(attributes):
        attr_status = []
        values_status = []
        parsed_idx = attribute_mapping.get(i, i)
        
        # Get expected values for this attribute
        expected_values = formatted_truth[i]
        
        # Normalize expected values for comparison
        expected_values_norm = [v.lower() for v in expected_values]
        expected_values_set = set(expected_values_norm)
        
        # Compare cell by cell
        for house_idx in range(expected_houses):
            parsed_value = parsed_lists[parsed_idx][house_idx].lower()
            expected_value = expected_values_norm[house_idx]
            
            is_correct = parsed_value == expected_value
            is_in_solution = parsed_value in expected_values_set
            
            if is_correct:
                correct_cells += 1
                values_status.append(True)  # Value is correct
            elif is_in_solution:
                values_status.append(True)  # Value is in solution but wrong position
            else:
                values_status.append(False)  # Value not in solution
            
            attr_status.append(is_correct)
        
        cell_status.append(attr_status)
        values_in_solution.append(values_status)
    
    # Calculate scores
    total_score = correct_cells / total_cells if total_cells > 0 else 0.0
    
    # Calculate values-in-solution score (percentage of values that exist in the solution)
    total_value_cells = total_cells
    correct_value_cells = sum(sum(row) for row in values_in_solution)
    values_score = correct_value_cells / total_value_cells if total_value_cells > 0 else 0.0
    
    # 4. Generate detailed feedback in markdown format
    feedback = []
    
    # Add text-only feedback
    text_feedback = []
    text_feedback.append("Text-only Solution Feedback:")
    
    for house_idx in range(expected_houses):
        house_feedback = [f"House {house_idx+1}:"]
        
        for attr_idx in range(len(attributes)):
            parsed_idx = attribute_mapping.get(attr_idx, attr_idx)
            is_correct = cell_status[attr_idx][house_idx]
            is_in_solution = values_in_solution[attr_idx][house_idx]
            
            parsed_value = parsed_lists[parsed_idx][house_idx]
            expected_value = formatted_truth[attr_idx][house_idx]
            
            status = "correct" if is_correct else "should be in another house" if is_in_solution else "no in this attribute"
            house_feedback.append(f" {attributes[attr_idx]}: {parsed_value} is {status}." if parsed_value else f" {attributes[attr_idx]}: No given answer.")
        
        text_feedback.append(" ".join(house_feedback))
    
    feedback.extend(text_feedback)

    detailed_feedback = "\n".join(feedback)
    
    # Return validation results with combined score (giving 70% weight to exact matches and 30% to values-in-solution)
    combined_score = 1.0 * total_score + 0.0 * values_score
    is_perfect = correct_cells == total_cells
    return is_perfect, combined_score, detailed_feedback

def compute_score(solution_str: str, ground_truth: Dict[str, Any]) -> Tuple[float, str]:
    """
    Compute the reward score for a zebra puzzle solution.
    
    Args:
        solution_str: The solution string from the model
        ground_truth: The ground truth data
    
    Returns:
        Tuple of (reward score, feedback)
    """

    repeat_penalty = 0.0
    
    # Split into paragraphs and look for duplicates
    paragraphs = [p.strip() for p in solution_str.split('\n\n') if p.strip()]
    unique_paragraphs = set(paragraphs)
    
    if len(paragraphs) > len(unique_paragraphs):
        duplicate_count = len(paragraphs) - len(unique_paragraphs)
        repeat_penalty = min(0.2, duplicate_count * 0.05)  # Max penalty of 0.2 (20%)
    
    # Parse the solution into lists
    parsed_lists = parse_markdown_solution(solution_str, ground_truth["attributes"])
    if parsed_lists is None:
        return 0.0, "No answer found, you should provide your answer in the format of <answer>...</answer>."
    
    # Validate the parsed lists
    is_valid, score, feedback = validate_parsed_lists(parsed_lists, ground_truth)
    
    final_score = max(0.0, score - repeat_penalty)
    
    return final_score, feedback