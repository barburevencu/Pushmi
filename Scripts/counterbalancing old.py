import pandas as pd
import itertools
import random
import copy
import numpy as np
import os
from collections import defaultdict

def starts_with_vowel(s):
    if not s:  # Handle empty string case
        return False
    
    vowels = "aeèéiouhHAEÉÉÈIOU" # Or just "aeiou" and use s[0].lower()
    return s[0].lower() in vowels

# --- CORE LOGIC & HELPER FUNCTIONS ---
def sort_list(pairs):
    """Sorts pairs of items, grouping canonical and reversed versions."""
    grouped = defaultdict(list)
    for pair in pairs:
        key = tuple(sorted(pair))
        grouped[key].append(pair)

    ordered_pairs = []
    for key in sorted(grouped):
        items = grouped[key]
        # Ensure a consistent order for pairs
        items.sort()
        ordered_pairs.extend(items)
    return ordered_pairs

def balance_extra_variables(n, repeats, mirror=True):
    """
    Generates balanced lists of 'A'/'B' sequences for experimental variables,
    with a constrained shuffle to prevent mirror versions from being paired.
    """
    base_sequences = [list(seq) for seq in itertools.product(['A', 'B'], repeat=repeats)]
    #return
    shuffled_versions = []
    seen_keys = set()
    while len(shuffled_versions) < n:
        shuffled = copy.deepcopy(base_sequences)
        random.shuffle(shuffled)
        # Use a string representation of the list to check for uniqueness
        key = str(shuffled)
        
        if key not in seen_keys:
            shuffled_versions.append(shuffled)
            seen_keys.add(key)

    if not mirror:
        return shuffled_versions

    # Create mirrored versions (swapping 'A' and 'B')
    swapped_versions = [
        [['B' if item == 'A' else 'A' for item in row] for row in version]
        for version in shuffled_versions
    ]

    # Create a lookup dictionary to easily find the mirror of any version.
    # The string representation, which is used for uniqueness, serves as a reliable key.
    mirror_lookup = {}
    for i in range(n):
        original_key = str(shuffled_versions[i])
        mirror_key = str(swapped_versions[i])
        mirror_lookup[original_key] = mirror_key
        mirror_lookup[mirror_key] = original_key

    # Combine all versions into a single list for shuffling.
    combined = shuffled_versions + swapped_versions
    
    max_attempts = 1000  # A safeguard to prevent an infinite loop.
    for _ in range(max_attempts):
        random.shuffle(combined)
        
        # Check if the shuffle is valid by ensuring no adjacent items are mirrors.
        is_valid = True
        # Iterate through the list in pairs (0,1), (2,3), etc.
        for i in range(0, len(combined), 2):
            
            item1_key = str(combined[i])
            item2_key = str(combined[i+1])
            
            # If item2 is the mirror of item1, the shuffle is invalid.
            if mirror_lookup[item1_key] == item2_key:
                is_valid = False
                break  # No need to check further; try a new shuffle.
        
        if is_valid:
            return combined  # A valid, constrained shuffle was found.

    # This part is reached only if a valid shuffle isn't found after many tries.
    raise RuntimeError(
        "Failed to generate a valid constrained shuffle after "
        f"{max_attempts} attempts. This may indicate an issue or a very "
        "low probability of finding a valid arrangement."
    )

def get_outcome(agent_assignment, central_shape, peripheral_position, movement):
    """Determines the outcome ('pushed' or 'pulled') based on trial conditions."""
    if central_shape == agent_assignment:
        return "pousse" if movement == peripheral_position else "tire"
    else:
        return "pousse" if movement != peripheral_position else "tire"

def get_ground_truth(agent, patient, outcome, change = None):
    if starts_with_vowel(agent):
        subject = f"L'{agent}" 
    elif agent in ["pentagone", "pigeon", "mouton", "marteau", "couteau"]:
        subject = f"Le {agent}"
    else:
        subject = f"La {agent}"

    if starts_with_vowel(patient):
        dir_object = f"l'{patient}" 
    elif patient in ["pentagone", "pigeon", "mouton", "marteau", "couteau"]:
        dir_object = f"le {patient}"
    else:
        dir_object = f"la {patient}"

    return f"{subject} {outcome} {dir_object}"

def get_test_sentence(agent, patient, outcome, change):
    """Generates the test sentence, applying changes if specified."""
    swap_map = {
        'agent': {'biche': 'dinde', 'dinde': 'biche', 'pigeon': 'mouton', 'mouton': 'pigeon', 'étoile': 'ellipse', 'pentagone': 'hexagone'},
        'patient': {'pince': 'lampe', 'lampe': 'pince', 'marteau': 'couteau', 'couteau': 'marteau', 'étoile': 'ellipse', 'pentagone': 'hexagone'},
        'verb': {'pousse': 'tire', 'tire': 'pousse'}
    }

    if pd.notna(change) and change in swap_map:
        if change == 'agent':
            agent = swap_map[change].get(agent, agent)
        elif change == 'patient':
            patient = swap_map[change].get(patient, patient)
        elif change == 'verb':
            outcome = swap_map[change].get(outcome, outcome)

    if starts_with_vowel(agent):
        subject = f"L'{agent}" 
    elif agent in ["pentagone", "pigeon", "mouton", "marteau", "couteau"]:
        subject = f"Le {agent}"
    else:
        subject = f"La {agent}"

    if starts_with_vowel(patient):
        dir_object = f"l'{patient}" 
    elif patient in ["pentagone", "pigeon", "mouton", "marteau", "couteau"]:
        dir_object = f"le {patient}"
    else:
        dir_object = f"la {patient}"

    return f"{subject} {outcome} {dir_object}"

def _generate_base_dataframe(shapes, agents, patients, extra_vars_config, change_counts, double=False):
    """
    Core refactored function to generate a counterbalanced DataFrame.
    This contains all the logic shared between the training and experiment scripts.
    """
    # 1. Generate all primary trial combinations
    shape_choice = sort_list(list(itertools.permutations(shapes, 2)))
    agent_assignment = ["shape1", "shape2"]
    assignment_order = ["symbol_first", "referent_first"]
    
    trial_combinations = list(itertools.product(agent_assignment, agents, patients, shape_choice, assignment_order))
    
    mappings = {
        'central_shape': {'A': 'shape1', 'B': 'shape2'},
        'peripheral_position': {'A': 'left', 'B': 'right'},
        'movement': {'A': 'left', 'B': 'right'},
        'correct_key': {'A': 'right', 'B': 'left'}
    }

    attempt = 0
    while True:
        # All existing code to generate df up to this point
        df = pd.DataFrame(trial_combinations, columns=[
            'agent_assignment', 'agent', 'patient', 'shape_pair', 'assignment_order'
        ])

        df['shape1'] = df['shape_pair'].apply(lambda x: x[0])
        df['shape2'] = df['shape_pair'].apply(lambda x: x[1])
        df = df.drop(columns=['shape_pair'])

        df['patient_assignment'] = df['agent_assignment'].apply(lambda x: 'shape2' if x == 'shape1' else 'shape1')
        df['label1'] = df.apply(lambda row: row['agent'] if row['agent_assignment'] == 'shape1' else row['patient'], axis=1)
        df['label2'] = df.apply(lambda row: row['patient'] if row['agent_assignment'] == 'shape1' else row['agent'], axis=1)
        
        df['shape_pair'] = df.apply(lambda row: "-".join(sorted([row['shape1'], row['shape2']])), axis=1)
        df['alphabet_order'] = df['shape1'] < df['shape2']

        df = df.sort_values(by=['shape_pair', 'agent_assignment', 'agent', 'patient', 'alphabet_order', 'assignment_order']).reset_index()

        # print(df.head(33))
        extra_variables = balance_extra_variables(
            n=extra_vars_config['n'], 
            repeats=extra_vars_config['repeats'], 
            mirror=extra_vars_config['mirror']
        )

        extra_df = pd.DataFrame(
            [row for sublist in extra_variables for row in sublist],
            columns=['central_shape', 'peripheral_position', 'movement', 'correct_key']
        )
        
        if len(df) != len(extra_df):
            raise ValueError(f"Row count mismatch: Main DataFrame has {len(df)} rows, but extra variables have {len(extra_df)} rows.")

        df = pd.concat([df, extra_df], axis=1)
        df = df.drop(columns=['alphabet_order', 'index'])

        if double:
            df_copy = df.copy(deep=True)
            for col in mappings.keys():
                df_copy[col] = df_copy[col].replace({'A': 'B', 'B': 'A'})
            df = pd.concat([df, df_copy], ignore_index=True)

        for col, mapping in mappings.items():
            df[col] = df[col].map(mapping)

        df['change'] = np.nan
        change_values = [k for k, v in change_counts.items() for _ in range(v)]
        
        random.shuffle(change_values)
        f_key_mask = df['correct_key'] == 'left'

        if f_key_mask.sum() == len(change_values):
            if 'change' not in df.columns or df['change'].dtype != 'object':
                df['change'] = df['change'].astype('object')

            df.loc[f_key_mask, 'change'] = change_values

        else:
            print(f"Warning: Number of change trials ({len(change_values)}) does not match available slots ({f_key_mask.sum()}). Change column may be incomplete.")
       
        df['change'] = df['change'].astype('object')
        df['outcome'] = df.apply(lambda row: get_outcome(row['agent_assignment'], row['central_shape'], row['peripheral_position'], row['movement']), axis=1)
        
        # FORCE EVEN BALANCING
        is_balanced = True

        for col in ['assignment_order', 'agent', 'patient', 'correct_key']:
            counts = df.groupby(col)['outcome'].value_counts().unstack(fill_value=0)
            if not counts['pousse'].equals(counts['tire']):
                is_balanced = False
                break
        
        attempt += 1

        if is_balanced:
            print(attempt)
            break

    # 6. Generate final columns
    #df['shape_pair'] = df.apply(lambda row: "-".join([row['shape1'], row['shape2']]), axis=1)
    df['ground_truth'] = df.apply(lambda row: get_ground_truth(row['agent'], row['patient'], row['outcome']), axis=1)
    df['test_sentence'] = df.apply(lambda row: get_test_sentence(row['agent'], row['patient'], row['outcome'], row['change']), axis=1)
    df['location_order'] = np.where(df['central_shape'] == 'shape1', 'shape1_first', 'shape2_first')

    # 7. Finalize by shuffling and adding a trial number
    df = df.sample(frac=1).reset_index(drop=True)
    df['trial_number'] = range(1, len(df) + 1)
    
    return df

# --- PUBLIC COUNTERBALANCING FUNCTIONS ---
def counterbalance_training():
    """Generates the counterbalanced design for the TRAINING phase."""
    df = _generate_base_dataframe(
        shapes=["pentagone", "étoile"],
        agents=["pigeon", "mouton"],
        patients=["marteau", "couteau"],
        extra_vars_config={'n': 2, 'repeats': 4, 'mirror': False},
        change_counts={"agent": 5, "patient": 5, "verb": 6}
    )

    # Add training-specific 'trial_type' column
    conditions = [
        df['trial_number'] <= 6,
        (df['trial_number'] > 6) & (df['trial_number'] <= 16)
    ]
    choices = ['training_no_labels', 'training_labels']
    df['trial_type'] = np.select(conditions, choices, default='training_no_animation')
    df['trial_block'] = 0

    # Modify 'training_no_labels' trials to use shape names instead of agent/patient names
    no_labels_mask = df['trial_type'] == 'training_no_labels'
    
    def generate_shape_sentence(row, base_function):
        agent_shape = row['shape1'] if row['agent_assignment'] == 'shape1' else row['shape2']
        patient_shape = row['shape2'] if row['agent_assignment'] == 'shape1' else row['shape1']
        return base_function(agent_shape, patient_shape, row['outcome'], row.get('change'))

    # Apply the sentence modifications
    df.loc[no_labels_mask, 'ground_truth'] = df[no_labels_mask].apply(lambda r: generate_shape_sentence(r, get_ground_truth), axis=1)
    df.loc[no_labels_mask, 'test_sentence'] = df[no_labels_mask].apply(lambda r: generate_shape_sentence(r, get_test_sentence), axis=1)
    
    # Replace labels with shape names for these specific trials
    df.loc[no_labels_mask, 'label1'] = df.loc[no_labels_mask, 'shape1']
    df.loc[no_labels_mask, 'label2'] = df.loc[no_labels_mask, 'shape2']

    return df

def counterbalance_experiment(double=False):
    """Generates the counterbalanced design for the main EXPERIMENT phase."""
    change_num = 64 if double else 32
    df = _generate_base_dataframe(
        shapes=["cercle", "carré", "croix", "losange"],
        agents=["biche", "dinde"],
        patients=["lampe", "pince"],
        extra_vars_config={'n': 6, 'repeats': 4, 'mirror': True},
        change_counts={"agent": change_num, "patient": change_num, "verb": change_num},
        double=double
    )
    df['trial_type'] = 'test'
    df['trial_block'] = (df.index // 32) + 1

    return df

def counterbalance(ID = 1, double=True):
    """
    Combines the training and experiment data into a single DataFrame,
    saves it, and returns the result.
    """
    # Get the individual dataframes
    # df_localizer_training = counterbalance_localizer(4, max_spacing = 30)
    # df_localizer = counterbalance_localizer(60)
    # df_training = counterbalance_training()
    df_experiment = counterbalance_experiment(double=double)
    df_experiment['ID'] = ID

    # Concatenate them, with training first
    # combined_df = pd.concat([df_localizer_training, df_localizer, df_training, df_experiment], ignore_index=True)
    # combined_df['ID'] = ID

    column_order = [
        'ID',
        'trial_type',
        'trial_block',
        'trial_number',
        'assignment_order', 
        'shape1', 'shape2', 'shape_pair',
        'agent_assignment', 'patient_assignment', 'label1', 'label2',
        'agent', 'patient',
        'central_shape', 'peripheral_position', 'location_order', 'movement', 'outcome',
        'ground_truth', 'change', 'test_sentence', 'correct_key'
    ]

    return df_experiment[column_order]

def counterbalance_localizer(n=64, max_spacing=30):
    """Generates the design for the localizer task, alternating between word and image trials."""
    def is_valid_correct_spacing(correct_series, max_spacing):
        correct_indices = correct_series[correct_series == 1.0].index.to_list()

        # 1. Check if there are any correct values
        if not correct_indices:
            return False

        # 2. Calculate all spacings
        # The first element is the index of the first correct value
        # The rest are differences between consecutive correct indices
        all_spacings = [correct_indices[0]] + [j - i for i, j in zip(correct_indices, correct_indices[1:])]

        # 3. Check the maximum spacing constraint for ALL calculated spacings
        # This ensures no spacing (including the initial one) exceeds max_spacing
        if not all(spacing <= max_spacing for spacing in all_spacings):
            return False

        # If all checks pass
        return True

    meanings = ["biche", "dinde", "lampe", "pince", "cercle", "carré", "croix", "losange"] 
    
    # Create lists for 'word' and 'image' trials
    word_trials = []
    for _ in range(n):
        for word in meanings:
            word_trials.append((word, "word"))
    
    image_trials = []
    for _ in range(n):
        for image in meanings:
            image_trials.append((image, "image"))

    max_attempts = 10000
    attempt = 1

    while attempt < max_attempts:
        random.shuffle(word_trials)
        random.shuffle(image_trials)

        # Interleave the trials
        interleaved_pairs = []
        min_len = len(word_trials)

        for i in range(min_len):
            interleaved_pairs.append(word_trials[i])
            interleaved_pairs.append(image_trials[i])
        
        df = pd.DataFrame(interleaved_pairs, columns=["meaning", "stimulus"])
            
        # Calculate ground_truth based on meaning (alternation implies changing meaning)
        df['ground_truth'] = df['meaning'].eq(df['meaning'].shift()).astype('float')
        df.loc[0, 'ground_truth'] = np.nan # First trial has no previous trial to compare to for 'same/different' meaning
        midpoint = len(df) // 2
        
        if n > 20:
            df.loc[midpoint, 'ground_truth'] = np.nan # First trial has no previous trial to compare to for 'same/different' meaning

        if is_valid_correct_spacing(df['ground_truth'], max_spacing):
            df['trial_type'] = 'localizer'
            df['trial_number'] = range(1, len(df) + 1)

            # Add trial_block: 1 for first half, 2 for second half
            df['trial_block'] = [0 if n < 20
                            else 1 if i < midpoint
                            else 2                 
                            for i in range(len(df))
                        ]

            # Add label1 and shape1 based on stimulus type
            df['label1'] = df.apply(lambda row: row['meaning'] if row['stimulus'] == 'word' else np.nan, axis=1)
            df['shape1'] = df.apply(lambda row: row['meaning'] if row['stimulus'] == 'image' else np.nan, axis=1)

            # Drop 'meaning' and 'stimulus' columns
            df = df.drop(columns=['meaning', 'stimulus'])

            # Add correct_key
            df['correct_key'] = df['ground_truth'].map({0.0: np.nan, 1.0: 'right'})

            return df
            
        attempt += 1

# --- SCRIPT EXECUTION EXAMPLE ---
if __name__ == '__main__':
    NUM_SUBJECTS = 1
    print(f"Generating counterbalancing files for {NUM_SUBJECTS} subjects...")
    
    all_dfs = []
    for i in range(1, NUM_SUBJECTS + 1):
        print(f"  - Generating data for subject {i}...")
        df = counterbalance(ID=i, double=False)
        all_dfs.append(df)
    
    # # Combine all individual DataFrames into one
    # full_df = pd.concat(all_dfs, ignore_index=True)
    
    # # Define the output path and ensure the directory exists
    # #output_path = os.path.expanduser("/home/neurostim/MEG_studies/Pushmi_Barbu/Randomization/counterbalancing.csv")
    output_path = os.path.expanduser("/Users/Barbu/Documents/Postdoc/Experiments/Pushmi/MEG/Randomization/cb.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the final combined DataFrame to a CSV file
    df.to_csv(output_path, index=False)
