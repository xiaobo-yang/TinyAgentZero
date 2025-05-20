"""
Logic Puzzle Generator
This script generates logic puzzles similar to Einstein's puzzle or zebra puzzle,
where players need to determine the correct position of each element in a table based on given clues.
"""

import random
import collections
import time
from typing import Literal, List, Set, Tuple, Callable



def update_range(wns: List[str], rns: List[List[Set[str]]], cmp: Callable):
    """
    Update the potential solution set for a single rule
    
    Parameters:
        wns: Some attribute values
        rns: Solution set of possible values for each position
        cmp: Rule function involving wns
        
    Returns:
        Boolean indicating whether the solution set was updated
    """
    changed = False
    
    # 1. Update ranges according to the mutually exclusive property rule: if an attribute is determined at one position, it cannot appear in other positions
    for rn in rns:
        # Record solutions that have only one possibility
        classified_words = set()
        for n_col, set_of_words in enumerate(rn):
            if len(set_of_words) == 1:
                classified_words.add(next(iter(set_of_words))) # Extract the element from the set, note that elements can't be directly accessed by index
        
        # Remove already determined items from candidates, update ranges, e.g., if house 1 is apple, then other houses cannot be apple
        word_to_cols = dict()
        for n_col, set_of_words in enumerate(rn):
            if len(set_of_words) != 1:
                prev_length = len(set_of_words)
                set_of_words.difference_update(classified_words) # Update ranges
                # Solutions for one position shouldn't appear in other positions, e.g., if house 1 is apple, then other houses can't be apple
                changed |= prev_length != len(set_of_words) # If some candidates were successfully removed, ranges changed, changed becomes True
                for word in set_of_words:
                    word_to_cols.setdefault(word, set()).add(n_col) # Add data to word_to_cols dictionary with key=word, value={n_col}, indicating this attribute (word) may appear in which houses (n_col)
        
        
        # After updating ranges in the previous round, some position candidates may be reduced, if a candidate reduces to 1 solution, then the answer is determined, update ranges again
        for word, cols in word_to_cols.items():
            if len(cols) == 1:
                x = rn[next(iter(cols))]
                if len(x) != 1:
                    x.clear()
                    x.add(word) # If a word only appears in one position, then that position must be this word, other candidates can be removed
                    changed = True

    # 2. Update ranges according to the current rule function: the rule function constrains positions, remove those that don't satisfy the constraints
    new_rns = [[{x for x in xs if x != wn} for xs in rn] for wn, rn in zip(wns, rns)] # Remove attributes in the answer from ranges, create a copy
    
    # wns are several attributes involved in a rule, now we need to check all position combinations in the current ranges that have these attributes
    # For example:
    # wns = ['apple', 'doctor']
    # rns = [
    #     # Possible positions for apple
    #     [{'apple','banana'}, {'apple','banana'}, {'banana'}],  # apple might be at position 0 or 1
    #     # Possible positions for doctor
    #     [{'teacher'}, {'doctor','teacher'}, {'doctor','teacher'}]  # doctor might be at position 1 or 2
    # ]
    # Returns: pairs = [[0,1], [1,1], [0,2], [1,2]]
    pairs = []
    for wn, rn in zip(wns, rns):
        new_pairs = []
        break_condition = True
        for cn, setn in enumerate(rn):
            if wn in setn:
                break_condition = False
                if not pairs:
                    pairs = [[]]
                for v in pairs:
                    new_pairs.append([*v, cn])
        pairs = new_pairs
        if break_condition: # If a position's solution set doesn't contain the standard answer, the puzzle has no solution, exit directly
            break
    
    # Remove pairs that don't satisfy the rules, update ranges
    for pair in pairs:
        if cmp(*pair):
            for nrn, cn, wn in zip(new_rns, pair, wns):
                nrn[cn].add(wn) # Add these attributes back to the copied new_ranges, those not added back are removed
    
    # new_ranges != ranges means some ranges have been removed, the solution set has changed, update and exit
    changed |= any(rn != new_rn for rn, new_rn in zip(rns, new_rns))
    if changed:
        for rn, new_rn in zip(rns, new_rns):
            for old, new in zip(rn, new_rn):
                old.intersection_update(new)
    
    return changed


def update_ranges(relations, ranges):
    """
    Update potential solution sets for multiple rules
    
    Parameters:
        relations: List of relations
        ranges: Solution set
        
    Returns:
        Boolean indicating whether any solution set was updated
    """
    changed = False
    for ins, wns, callable_object, *_ in relations: # wns are the specific attribute names in the original answer involved in the rule
        changed |= update_range(wns, [ranges[i] for i in ins], callable_object) # rns=[ranges[i] for i in ins], the current potential answers for these attribute types
    return changed

def get_rules(m_objects: int,
              level: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
              except_flag: bool = True):
    """
    Set up candidate rule sets according to difficulty level
    Each element is a 3-tuple or 4-tuple:
    3-tuple: (n_args, cmp_function, str_variants)
    4-tuple: (n_args, cmp_function, str_variants, except_flag)
        - n_args: Number of attributes involved in the rule
        - cmp_function: Logical judgment function abstracted from the rule
        - str_variants: Text description placeholder template for the rule, can be filled with actual names using .format method. Can have multiple equivalent expressions, a random one will be chosen when generating rule text to provide diversity
        - (optional) except_flag: Some judgment conditions may produce trivial rules, this flag skips them directly, such as "Food:apple != Food:banana", "Job:doctor != Job:teacher"
    """
    rules_for_relations = [
        (2, lambda j1, j2: j1 == j2,
        ['The person who likes {1} as their {0} is the same as the person who likes {3} as their {2}', 'The person who likes {3} as their {2} is the same as the person who likes {1} as their {0}',
        'The person who likes {1} as their {0} is the same person who likes {3} as their {2}', 'The person who likes {3} as their {2} is the same person who likes {1} as their {0}'], except_flag),
        (2, lambda j1, j2: j1 + 1 == j2,
        ['The person who likes {1} as their {0} is on the left of the person who likes {3} as their {2}, no one in between',
        'The person who likes {1} as their {0} is located on the left of the person who likes {3} as their {2}, no one in between']),
        (2, lambda j1, j2: j1 == j2 + 1,
        ['The person who likes {1} as their {0} is on the right of the person who likes {3} as their {2}, no one in between',
        'The person who likes {1} as their {0} is located on the right of the person who likes {3} as their {2}, no one in between']),
        (1, lambda j: j == 0,
        ['The person who likes {1} as their {0} is on the far left',
        'The person who likes {1} as their {0} is located on the far left']),
        (1, lambda j: j == m_objects - 1,
        ['The person who likes {1} as their {0} is on the far right',
        'The person who likes {1} as their {0} is located on the far right']),
        ] + (m_objects % 2 != 0) * [(1, lambda j: j == m_objects // 2, ['The person who likes {1} as their {0} is in the middle', 'The person who likes {1} as their {0} is located in the center'])]
    if level >= 2:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j2 + 1 == j1 == j3 - 1 or j3 + 1 == j1 == j2 - 1,
            ['The person who likes {1} as their {0} is between the person who likes {3} as their {2} and the person who likes {5} as their {4}', 
            'The person who likes {1} as their {0} is between the person who likes {5} as their {4} and the person who likes {3} as their {2}']),
        ]
    if level >= 3:
        rules_for_relations += [
            (2, lambda j1, j2: j1 == j2 - 1 or j1 == j2 + 1,
            ['The person who likes {1} as their {0} is either on the left or right of the person who likes {3} as their {2}, no on in between']),
            (1, lambda j1, last_index=m_objects - 1: j1 == 0 or j1 == last_index,
            ['The person who likes {1} as their {0} is either on the far left or far right']),
        ]
    if level >= 4:
        rules_for_relations += [
            (1, lambda j1: (j1 + 1) % 2 != 0, ['The person who likes {1} as their {0} is in an odd-numbered position']),
            (1, lambda j1: (j1 + 1) % 2 == 0, ['The person who likes {1} as their {0} is in an even-numbered position']),
        ]
    if level >= 5:
        rules_for_relations += [
            (2, lambda j1, j2: j1 < j2, ['The person who likes {1} as their {0} is somewhere to the left of the person who likes {3} as their {2}, possibly with others in between']),
            (2, lambda j1, j2: j1 > j2, ['The person who likes {1} as their {0} is somewhere to the right of the person who likes {3} as their {2}, possibly with others in between']),
        ]
    if level >= 6:
        rules_for_relations += [
            (2, lambda j1, j2: j1 != j2, ['The person who likes {1} as their {0} is not the person who likes {3} as their {2}', 'The person who likes {1} as their {0} and the person who likes {3} as their {2} are not the same person'
                                        'The person who likes {3} as their {2} is not the person who likes {1} as their {0}', 'The person who likes {3} as their {2} and the person who likes {1} as their {0} are not the same person'], except_flag),
        ]
    if level >= 7:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j2 < j1 < j3 or j3 < j1 < j2,
            ['The person who likes {1} as their {0} is somewhere between the person who likes {3} as their {2} and the person who likes {5} as their {4}',
            'The person who likes {1} as their {0} is somewhere between the person who likes {5} as their {4} and the person who likes {3} as their {2}']),
        ]
    if level >= 8:
        rules_for_relations += [
            (2, lambda j1, j2: j1 >= j2, ['The person who likes {1} as their {0} is to the right of the person who likes {3} as their {2}, possibly with others in between; Or The person who likes {1} as their {0} is the same as the person who likes {3} as their {2}'], except_flag),
            (2, lambda j1, j2: j1 <= j2, ['The person who likes {1} as their {0} is to the left of the person who likes {3} as their {2}, possibly with others in between; Or The person who likes {1} as their {0} is the same as the person who likes {3} as their {2}'], except_flag),
        ]
    if level >= 9:
        rules_for_relations += [
            (2, lambda j1, j2: j1 % 2 != j2 % 2,
            ['The person who likes {1} as their {0} and the person who likes {3} as their {2} are in positions with different parity',
            'The person who likes {3} as their {2} and the person who likes {1} as their {0} are in positions with different parity'], except_flag),
            (2, lambda j1, j2: j1 % 2 == j2 % 2,
            ['The person who likes {1} as their {0} and the person who likes {3} as their {2} are in positions with the same parity',
            'The person who likes {3} as their {2} and the person who likes {1} as their {0} are in positions with the same parity'], except_flag),
        ]
    if level >= 10:
        rules_for_relations += [
            (3, lambda j1, j2, j3: (j1 == j2 and j1 != j3) or (j1 != j2 and j1 == j3),
            ['The person who likes {1} as their {0} is either the person who likes {3} as their {2}, or the person who likes {5} as their {4}, but not both',
            'The person who likes {1} as their {0} is either the person who likes {5} as their {4}, or the person who likes {3} as their {2}, but not both'], except_flag),
            (3, lambda j1, j2, j3: (j1 == j2 and j2 != j3) or (j1 != j2 and j2 == j3),
            ['The person who likes {1} as their {0} is the person who likes {3} as their {2}, or the person who likes {3} as their {2} is the person who likes {5} as their {4}, but not both',
            'The person who likes {3} as their {2} is the person who likes {5} as their {4}, or the person who likes {1} as their {0} is the person who likes {3} as their {2}, but not both'], except_flag),
        ]
    if level >= 11:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j1 == j2 or j1 == j3,
            ['The person who likes {1} as their {0} is the person who likes {3} as their {2}, or the person who likes {1} as their {0} is the person who likes {5} as their {4}, or both statements are true',
            'The person who likes {1} as their {0} is the person who likes {5} as their {4}, or the person who likes {1} as their {0} is the person who likes {3} as their {2}, or both statements are true'], except_flag),
            (3, lambda j1, j2, j3: j1 == j2 or j2 == j3,
            ['The person who likes {1} as their {0} is the person who likes {3} as their {2}, or the person who likes {3} as their {2} is the person who likes {5} as their {4}, or both statements are true',
            'The person who likes {3} as their {2} is the person who likes {5} as their {4}, or the person who likes {1} as their {0} is the person who likes {3} as their {2}, or both statements are true'], except_flag),
        ]
    if level >= 12:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j1 != j2 or j1 != j3,
            ['The person who likes {1} as their {0} is not the person who likes {3} as their {2}, or the person who likes {1} as their {0} is not the person who likes {5} as their {4}, or both statements are true',
            'The person who likes {1} as their {0} is not the person who likes {5} as their {4}, or the person who likes {1} as their {0} is not the person who likes {3} as their {2}, or both statements are true'], except_flag),
            (3, lambda j1, j2, j3: j1 != j2 or j2 != j3,
            ['The person who likes {1} as their {0} is not the person who likes {3} as their {2}, or the person who likes {3} as their {2} is not the person who likes {5} as their {4}, or both statements are true',
            'The person who likes {3} as their {2} is not the person who likes {5} as their {4}, or the person who likes {1} as their {0} is not the person who likes {3} as their {2}, or both statements are true'], except_flag),
        ]
    if level >= 13:
        rules_for_relations.pop(0)  # Remove 'same'
    if level >= 14:
        rules_for_relations.pop(0)  # Remove 'on the left'
        rules_for_relations.pop(0)  # Remove 'on the right'
    if level >= 15:
        rules_for_relations.pop(0)  # Remove 'on the far left'
        rules_for_relations.pop(0)  # Remove 'on the far right'
        if m_objects % 2 != 0:
            rules_for_relations.pop(0)  # Remove 'in the middle'
    if level >= 16:
        rules_for_relations.pop(0)  # Remove 'between'
    if level >= 17:
        rules_for_relations.pop(0)  # Remove 'left or right'
        rules_for_relations.pop(0)  # Remove 'far left or far right'
    if level >= 18:
        rules_for_relations.pop(0)  # Remove 'odd position'
        rules_for_relations.pop(0)  # Remove 'even position'
    if level >= 19:
        rules_for_relations.pop(0)  # Remove 'somewhere to the left'
        rules_for_relations.pop(0)  # Remove 'somewhere to the right'
    if level >= 20:
        rules_for_relations.pop(0)  # Remove 'not'
    
    return rules_for_relations


# Attribute names
kinds_dict = {
    "Nationality": {  # Nationality
        "American", "Argentinian", "Australian", "Brazilian", "British",
        "Canadian", "Chinese", "Colombian", "Dutch", "Egyptian",
        "French", "German", "Indian", "Indonesian", "Italian",
        "Japanese", "Malaysian", "Mexican", "Nigerian", "Pakistani",
        "Polish", "Russian", "Spanish", "Thai", "Turkish",
    },
    "Favorite Food": {  # Food
        "Apple", "Apricot", "Artichoke", "Asparagus", "Avocado",
        "Banana", "Blueberry", "Broccoli", "Cabbage", "Carrot",
        "Cauliflower", "Cherry", "Corn", "Cranberry", "Cucumber",
        "Eggplant", "Garlic", "Grapefruit", "Grape", "Kale",
        "Kiwi", "Lemon", "Lettuce", "Lime", "Mango",
        "Nectarine", "Onion", "Orange", "Papaya", "Peach",
        "Pear", "Pea", "Pepper", "Pineapple", "Plum",
        "Pomegranate", "Potato", "Pumpkin", "Radish", "Raspberry",
        "Spinach", "Strawberry", "Tomato", "Watermelon", "Zucchini",
    },
    "Pet": {  # Pet
        "Bird", "Cat", "Chinchilla", "Dog", "Ferret",
        "Fish", "Frog", "Goat", "Goldfish", "Guinea Pig",
        "Hamster", "Hedgehog", "Horse", "Lizard", "Mouse",
        "Pony", "Rabbit", "Rat", "Snake", "Turtle",
    },
    "Profession": {  # Profession
        "Accountant", "Analyst", "Architect", "Bartender", "Chef",
        "Coach", "Dancer", "Designer", "Doctor", "Dressmaker",
        "Electrician", "Engineer", "Entrepreneur", "Firefighter", "Fisherman",
        "Freelancer", "Journalist", "Lawyer", "Librarian", "Manager",
        "Mechanic", "Musician", "Nurse", "Caregiver", "Photographer",
        "Pilot", "Police Officer", "Project Manager", "Scientist", "Security Guard",
        "Social Worker", "Software Developer", "Teacher", "Video Producer", "Writer",
    },
    "Favorite Drink": {  # Beverage
        "7UP", "Almond Milk", "Coffee", "Cola", "Fanta",
        "Hot Chocolate", "Iced Tea", "Juice", "Lemonade", "Milk",
        "Mirinda", "Soy Milk", "Sprite", "Tea", "Water",
    },
    "Preferred Transportation": {  # Transportation
        "Airplane", "Bicycle", "Boat", "Bus", "Car",
        "Helicopter", "Jet Ski", "Motorcycle", "Quad", "Roller",
        "Scooter", "Ship", "Skateboard", "Snowmobile",
        "Subway", "Taxi", "Train", "Tram", "Tricycle", "Van",
    },
    "Favorite Music Genre": {  # Music Genre
        "Ambient", "Blues", "Classical", "Country", "Dubstep",
        "Disco", "Drum and Bass", "Electronic", "Folk", "Funk",
        "Gospel", "Hip Hop", "House", "Indie", "Jazz",
        "Metal", "Pop", "Punk", "R&B", "Reggae",
        "Rock", "Salsa", "Soul", "Techno", "Trance",
    },
    "Favorite Movie Genre": {  # Movie Genre
        "Action", "Adventure", "Animation", "Comedy", "Crime",
        "Disaster", "Documentary", "Drama", "Epic", "Family",
        "Fantasy", "Horror", "Martial Arts", "Musical", "Mystery",
        "Romance", "Satire", "Science", "Sports", "Spy",
        "Superhero", "Thriller", "Time Travel", "Western", "Zombie",
    },
    "Favorite Sport": {  # Sport
        "Badminton", "Baseball", "Basketball", "Biathlon", "Climbing",
        "Cricket", "Cycling", "Golf", "Handball", "Hockey",
        "Lacrosse", "Parkour", "Rowing", "Rugby", "Sailing",
        "Skateboarding", "Skiing", "Snowboarding", "Soccer", "Surfing",
        "Swimming", "Tennis", "Volleyball", "Water Polo", "Weightlifting",
    },
    "Hobby": {  # Hobby
        "Baking", "Board Games", "Camping", "Card Games", "Chess",
        "Collecting", "Cooking", "Dancing", "Drawing", "Filmmaking",
        "Fishing", "Gardening", "Hiking", "Magic", "Photography",
        "Puzzles", "Reading", "Rock Climbing", "Singing", "Skydiving",
        "Sudoku", "Traveling", "Video Games", "Woodworking", "Writing",
    }
}


def generate_puzzle(table: List[List[str]], *,
                    level: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                    minimal_conditions: bool = False, max_seconds_for_minimizing: float = None,
                    tries: int = 10):
    """
    Generate a logic puzzle
    
    Parameters:
        table: Table data, representing the prepared answer
        level: Rule difficulty level (1-20)
        minimal_conditions: Whether to minimize the number of rules
        max_seconds_for_minimizing: Time limit for execution
        tries: Number of attempts
        
    Returns:
        List of clues for the puzzle
    """
    if level not in range(1, 20 + 1):
        raise ValueError('level must be >= 1 and <= 20')

    # Remove the column of attribute names, keep only specific attributes
    table_wo_left = [row[1:] for row in table] # (n_attributes * m_objects)
    n_attributes = len(table_wo_left) # Number of attributes
    m_objects = len(table_wo_left[0]) # Number of houses

    if level >= 19 and m_objects == 2:
        raise ValueError('For levels >= 19, the number of objects is too small')
    elif m_objects <= 1:
        raise ValueError('m_objects must be >= 2')
    elif n_attributes <= 0:
        raise ValueError('n_attributes must be >= 1')


    # Rule pool
    rules_for_relations = get_rules(m_objects, level)
    
        
    # Control puzzle generation
    is_minimized = False
    time_elapsed = False
    min_relations = None
    while True:
        ranges = [[set(table_wo_left[i]) for _ in range(len(table_wo_left[i]))] for i in range(len(table_wo_left))]
        # After giving the answer table in advance, generate some rules based on the answer, stored in the relations list
        # ranges is used to check whether the current rules in relations can uniquely determine a solution
        # ranges represents the possible attribute choices for each position
        # - First dimension (size n_attributes): Attribute category (e.g., food, profession, movie genre, transportation)
        # - Second dimension (size m_objects): Position (1st, 2nd, 3rd... position)
        # - Third dimension: Set containing all possible values for that attribute at that position
        # An initial ranges example when n_attributes = 4, m_objects = 3:
        # [
        #     [{'banana', 'lime', 'tomato'}, {'banana', 'lime', 'tomato'}, {'banana', 'lime', 'tomato'}],

        #     [{'doctor', 'freelancer', 'pilot'}, {'doctor', 'freelancer', 'pilot'}, {'doctor', 'freelancer', 'pilot'}],

        #     [{'comedy', 'sports', 'western'}, {'comedy', 'sports', 'western'}, {'comedy', 'sports', 'western'}],

        #     [{'bus', 'train', 'van'}, {'bus', 'train', 'van'}, {'bus', 'train', 'van'}]
        # ]
        # ranges gradually shrinks with iteration, and finally if each position corresponds to only one attribute, that's the unique solution. If a position is empty, there's no solution.
        relations = list()
        fail = False
        # The inner loop is controlled by fail, setting fail=True will restart
        # Each restart of the inner loop will generate a new set of relations
        while not fail:

            # Determine if the current candidate state (ranges) has already solved the problem
            needs_clarification = list()
            no_solutions = False
            solved = True
            for i, rng in enumerate(ranges):
                for j, rs in enumerate(rng):
                    # No answer exists for a position, problem has no solution
                    if len(rs) == 0:
                        no_solutions = True
                        solved = False
                        break
                    # Multiple answers exist for a position, needs further analysis
                    elif len(rs) > 1:
                        solved = False
                        needs_clarification.append((i, j)) # Record attribute positions that may have multiple potential answers, i represents attribute type, j represents house position
                if no_solutions:
                    break
            
            # Conditions to enter this branch:
            # 1. First entry: Since min_relations is initially None, this branch is first entered only when solved=True, when all attributes in ranges have only one candidate, len(rs) == 1
            # 2. Subsequent entry: min_relations is now defined, so besides ranges being solved early, it could also be that the current number of generated rules exceeds the number in a previously generated rule set
            # We always maintain the smallest possible relations rule set as the final output rules
            if solved or min_relations is not None and len(relations) >= len(min_relations):
                tries -= 1
                if min_relations is None or len(relations) < len(min_relations):
                    min_relations = relations
                if tries > 0:
                    fail = True
                    continue
            
            if tries <= 0:
                relations = min_relations
                if not minimal_conditions:
                    break

                # minimal_conditions controls whether to further reduce the rule set to the minimum
                # If minimal_conditions=False, directly use the smallest relations set obtained after tries rounds of attempts as the final output
                number_of_relations_min = len(relations)
                number_of_relations_before = len(relations)
                start_time = time.monotonic()

                # Maintain a stack of rules for breadth-first search until the smallest rule set relations is found
                main_q = collections.deque([relations])
                while main_q:
                    
                    current_relations = main_q.popleft()

                    # Check each rule in current_relations to see if removing it would result in no solution
                    # The advantage of BFS is that if a parent relations has multiple solutions, all its further reduced child relations also have multiple solutions, so they don't need to be checked, saving computation
                    for k in range(len(current_relations)):
                        # Initialize a new new_ranges
                        new_ranges = [[set(table_wo_left[i]) for _ in range(len(table_wo_left[i]))]
                                        for i in range(len(table_wo_left))]
                        
                        # Remove one by one
                        new_relations = current_relations.copy()
                        new_relations.pop(k)
                        
                        # Update new_ranges according to new_relations
                        changed = True
                        while changed:
                            changed = update_ranges(new_relations, new_ranges) 


                        # Verify whether the current rule set new_relations can yield a unique solution
                        # Since the solution set may not be unique after applying rules, to efficiently traverse all potential combinations in the solution set, maintain a stack of ranges for breadth-first search (BFS)
                        # This will break down all positions with multiple possible solutions, for example: 
                        # Breaking down position (0,2):
                        # temp_ranges1 = [
                        #     [{'apple'}, {'banana'}, {'apple'}],  # Assuming position 3 is apple
                        #     [{'doctor'}, {'teacher'}, {'doctor','teacher'}]
                        # ]
                        # temp_ranges = [
                        #     [{'apple'}, {'banana'}, {'banana'}],
                        #     [{'doctor'}, {'teacher'}, {'doctor','teacher'}]
                        # ]
                        # And each temp-ranges can be similarly broken down to get all combinations of solutions
                        # The advantage of BFS is that if a parent ranges has no solution, then all its further split child ranges also have no solution, so they don't need to be checked, saving computation
                        q = collections.deque([new_ranges])
                        possible_solutions = []
                        while q:
                            current_ranges = q.popleft()

                            no_solutions = False
                            solved = True
                            
                            # Check if the solution set is already unique
                            for rng in current_ranges:
                                for rs in rng:
                                    if len(rs) == 0: # No solution
                                        no_solutions = True
                                        solved = False
                                        break
                                    elif len(rs) > 1: # Still has multiple solutions
                                        solved = False
                                if no_solutions or not solved:
                                    break
                            
                            # If there's no solution, the current new_relations doesn't need further consideration
                            if no_solutions: 
                                continue
                            
                            if solved:
                                if current_ranges not in possible_solutions:
                                    possible_solutions.append(current_ranges)
                                    if len(possible_solutions) >= 2: # Has multiple solutions, indicating this rule is too weak, no need to check further
                                        break
                                continue
                            

                            for n_group, rng in enumerate(current_ranges):
                                founded = False
                                for n_x, rs in enumerate(rng):
                                    if len(rs) > 1: # In current_ranges, an attribute may have multiple solutions
                                        founded = True
                                        for r in rs:
                                            new_ranges = [[x.copy() for x in row] for row in current_ranges]
                                            new_ranges[n_group][n_x] = {r} # Keep only this solution
                                            
                                            # Update new_ranges that keeps only one solution according to the current rule new_relations, they represent fixing that multi-solution position, listing all possibilities
                                            # For example:
                                            # new_ranges = [
                                            #     [{'apple'}, {'banana'}, {'apple','banana'}],
                                            #     [{'doctor'}, {'teacher'}, {'doctor','teacher'}]
                                            # ]
                                            # Contains the following possible solution combinations:
                                            # temp_ranges = [
                                            #     [{'apple'}, {'banana'}, {'apple'}],  # Assuming position 3 is apple
                                            #     [{'doctor'}, {'teacher'}, {'doctor','teacher'}]
                                            # ]
                                            # temp_ranges = [
                                            #     [{'apple'}, {'banana'}, {'banana'}],
                                            #     [{'doctor'}, {'teacher'}, {'doctor','teacher'}]
                                            # ]
                                            # Verify these situations one by one to see if they satisfy new_relations
                                            changed = True
                                            while changed:
                                                changed = update_ranges(new_relations, new_ranges)
                                            
                                            q.appendleft(new_ranges)
                                        break
                                if founded: # Stop once such a ranges is found
                                    break
                        
                        # Finally, only new_relations with a unique solution is a reasonable rule set, then check if it reduced the number of rules, and ultimately keep the relations with the minimum number of rules
                        if len(possible_solutions) == 1:
                            number_of_relations_after = len(new_relations)
                            if number_of_relations_min > number_of_relations_after:
                                number_of_relations_min = number_of_relations_after
                                relations = new_relations
                                main_q.append(new_relations)
                        
                        # Force termination if timeout
                        if max_seconds_for_minimizing is not None and \
                                time.monotonic() >= start_time + max_seconds_for_minimizing:
                            time_elapsed = True
                            break
                    if time_elapsed:
                        break
                
                # Whether the minimum constraint set has been found
                is_minimized = number_of_relations_min < number_of_relations_before or not time_elapsed
                break

            if no_solutions or not needs_clarification:
                fail = True
                continue
            
            # Start analyzing from attribute positions with multiple potential answers
            i, j = item = random.choice(needs_clarification)
            next2_i, next2_j = None, None
            if level >= 2 and len(needs_clarification) > 1:
                needs_clarification.remove(item)
                next2_i, next2_j = random.choice(needs_clarification)

            neighbours = [] # All neighbors of position (i,j), including other attributes of house j, and all attributes of houses j-1 and j+1
            right_neighbours = [] # All neighbors of position (i,j), only including all attributes of house j+1
            for dj in range(-1, 1 + 1):
                if not (0 <= j + dj < m_objects):
                    continue
                for new_i in range(0, n_attributes):
                    if new_i == i and dj == 0:
                        continue
                    new_item = (new_i, j + dj)
                    neighbours.append(new_item)
                    if level >= 2 and dj == 1:
                        right_neighbours.append(new_item)
            if not neighbours:
                continue

            next_i, next_j = random.choice(neighbours)
            if level >= 2 and next2_i is None and right_neighbours:
                next2_i, next2_j = random.choice(right_neighbours)

            # Randomly select some objects from the current table to check which rules they satisfy
            permutations3 = [
                ((i, j), (next_i, next_j), (next2_i, next2_j)), ((i, j), (next2_i, next2_j), (next_i, next_j)),
                ((next_i, next_j), (i, j), (next2_i, next2_j)), ((next_i, next_j), (next2_i, next2_j), (i, j)),
                ((next2_i, next2_j), (i, j), (next_i, next_j)), ((next2_i, next2_j), (next_i, next_j), (i, j))
            ] if next2_i is not None else []
            permutations2 = [
                ((i, j), (next_i, next_j)), ((next_i, next_j), (next2_i, next2_j)), ((i, j), (next2_i, next2_j)),
                ((next_i, next_j), (i, j)), ((next2_i, next2_j), (next_i, next_j)), ((next2_i, next2_j), (i, j)),
            ] if next2_i is not None else [
                ((i, j), (next_i, next_j)), ((next_i, next_j), (i, j))
            ]


            # Traverse the pre-given rule set and select the rules that the randomly selected objects satisfy
            possible_variants = []
            # Elements are 4-tuples: (number of objects the rule compares, positions of objects satisfying the rule, rule function, rule description text)
            for (n_args, cmp_function, str_variants, *flags) in rules_for_relations:
                if n_args == 3:
                    for items in permutations3:
                        (ti, tj), (t_next_i, t_next_j), (t_next2_i, t_next2_j) = items
                        if flags and flags[0] and (ti == t_next_i or ti == t_next2_i or t_next_i == t_next2_i):
                            continue
                        if cmp_function(tj, t_next_j, t_next2_j):
                            possible_variants.append((n_args, items, cmp_function, random.choice(str_variants)))
                elif n_args == 2:
                    for items in permutations2:
                        (ti, tj), (t_next_i, t_next_j) = items
                        if flags and flags[0] and ti == t_next_i:
                            continue
                        if cmp_function(tj, t_next_j):
                            possible_variants.append((n_args, items, cmp_function, random.choice(str_variants)))
                elif n_args == 1 and cmp_function(j):
                    possible_variants.append((n_args, [(i, j)], cmp_function, random.choice(str_variants)))
            if not possible_variants:
                continue


            # From the satisfied rules, randomly select one as the rule text for the final selection of this round, add it to the rule pool
            n_args, list_of_ij, cmp_function, string_format = random.choice(possible_variants)
            list_for_format = []
            ins, wns = [], []
            for i, j in list_of_ij:
                list_for_format.extend([table[i][0], table_wo_left[i][j]])
                ins.append(i) # ins is the attribute category number
                wns.append(table_wo_left[i][j]) # wns is the specific attribute name, corresponding to several attributes involved in this rule
            relations.append((ins, wns, cmp_function, string_format.format(*list_for_format)))


            # According to the selected rule pool relations, exclude some ranges
            changed = True
            while changed:
                changed = update_ranges(relations, ranges) # Update ranges until the updated ranges no longer change compared to before
            

        if not fail:
            if minimal_conditions and not is_minimized and not time_elapsed:
                continue
            break

    premises = [t[-1] for t in relations]
    random.shuffle(premises)
    return premises

def generate_puzzle_data(n_attributes=2, m_objects=3, level=5, minimal_conditions=True):
    """
    Generate a puzzle data
    """
    # Define various attributes and their possible values
    kinds = sorted(kinds_dict)

    # Check
    assert n_attributes <= len(kinds_dict),\
        f'Insufficient number of attributes: Actual {len(kinds_dict)}, Expected {n_attributes}'
    assert all(m_objects <= len(v) for k, v in kinds_dict.items()), 'Insufficient number of objects: ' +\
        f'Actual {next(f"{k}={len(v)}" for k, v in kinds_dict.items() if m_objects > len(v))}, Expected {m_objects}'

    # Randomly generate table
    chosen_kinds = sorted(random.sample(kinds, k=n_attributes))
    table = [[kind] + random.sample(sorted(kinds_dict[kind]), k=m_objects) for kind in chosen_kinds]
    return table

def main(n_attributes=2, m_objects=3, level=5, minimal_conditions=True):
    """
    Main function - Generate and display the puzzle
    """
    # Define various attributes and their possible values
    kinds = sorted(kinds_dict)

    # Check
    assert n_attributes <= len(kinds_dict),\
        f'Insufficient number of attributes: Actual {len(kinds_dict)}, Expected {n_attributes}'
    assert all(m_objects <= len(v) for k, v in kinds_dict.items()), 'Insufficient number of objects: ' +\
        f'Actual {next(f"{k}={len(v)}" for k, v in kinds_dict.items() if m_objects > len(v))}, Expected {m_objects}'

    # Randomly generate table
    chosen_kinds = sorted(random.sample(kinds, k=n_attributes))
    table = [[kind] + random.sample(sorted(kinds_dict[kind]), k=m_objects) for kind in chosen_kinds]

    print('.:: Puzzle ::.')
    print(f'There are {m_objects} houses, numbered from 1 to {m_objects}. Each house has a different owner, and each person has {n_attributes} characteristics. These possible characteristics are as follows:')
    for row in table:
        print(f"{row[0]}:", ', '.join(sorted(row[1:])))
    print(f'Now I will provide you with some clues. Please deduce which characteristics each house owner has based on these clues. The clues are as follows:')
    t1 = time.monotonic()
    premises = generate_puzzle(table, level=level, minimal_conditions=minimal_conditions, max_seconds_for_minimizing=30)
    t2 = time.monotonic()
    indent = len(str(len(premises)))
    for i, premise in enumerate(premises, 1):
        i = str(i).rjust(indent)
        print(f"{i}. {premise}")
    print(f"Time used: {t2 - t1:.6f} seconds")
    return table


if __name__ == "__main__":
    data = main(3,4,20)
    print('\n.:: Answer ::.')
    print(data)