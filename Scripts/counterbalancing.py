import random, csv
from constants import *
from itertools import groupby, pairwise, permutations, product, combinations

def shuffled_copies(lst, n):
    """Return n unique shuffled versions of lst."""
    copies, seen = [], set()
    while len(copies) < n:
        new_list = lst[:]
        random.shuffle(new_list)
        sig = tuple(tuple(x) if isinstance(x, list) else x for x in new_list)
        if sig not in seen:
            seen.add(sig)
            copies.append(new_list)
    return copies

def mirror(obj, elem1='A', elem2='B'):
    """Swap elem1<->elem2 anywhere inside arbitrarily nested dict/list/tuple."""
    if isinstance(obj, dict):
        return {k: mirror(v, elem1, elem2) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(mirror(x, elem1, elem2) for x in obj)
    return elem2 if obj == elem1 else elem1 if obj == elem2 else obj

def balance_vars(n_vars, n_groups, mirrored):
    """Generate balanced A/B factor combinations for counterbalancing."""
    full_factor = [list(prod) for prod in product(['A', 'B'], repeat=n_vars)]
    shuffled = shuffled_copies(full_factor, n_groups)
    if not mirrored:
        return shuffled
    combined = shuffled + mirror(shuffled, "A", "B")
    for _ in range(10000):
        random.shuffle(combined)
        if all(combined[i] != mirror(combined[i + 1]) for i in range(0, len(combined), 2)):
            return combined

def is_balanced(trials, cols=('assignment_order', 'agent', 'patient', 'correct_key', 'central_shape')):
    """Return True if each column level has equal 'pousse' and 'tire' outcomes."""
    for col in cols:
        for _, group in groupby(sorted(trials, key=lambda t: t[col]), key=lambda t: t[col]):
            pousse, tire = 0, 0
            for trial in group:
                pousse += (trial['outcome'] == 'pousse')
                tire += (trial['outcome'] == 'tire')
            if pousse != tire:
                return False
    return True

def map_AB(trials, mappings):
    """Map A/B fields to concrete values in place."""
    for trial in trials:
        for col, mapping in mappings.items():
            if col in trial:
                trial[col] = mapping.get(trial[col], trial[col])

def add_numbers(trials, start_from=1):
    """Assign block_number and trial_number to trials."""
    for i, trial in enumerate(trials):
        trial["block_number"] = (i // BLOCK_SIZE) + start_from
        trial["trial_number"] = (i % BLOCK_SIZE) + 1

def sort_key(s):
    """Return a consistent key for a shape pair 'A-B' or 'B-A'."""
    return "-".join(sorted(s.split("-")))

def sort_trials(trials):
    """Sort trials by multiple keys."""
    trials.sort(key=lambda d: (d['shape_pair'], d["agent_shape"], d["agent"], d["patient"],
                                d['shape1'] + "-" + d['shape2'], d["assignment_order"]))

def article(noun):
    """Return the correct French definite article for a noun."""
    if noun[0].lower() in "aeiouhyéè":
        return f"L'{noun}"
    return f"{'Le' if noun in MASC_NAMES else 'La'} {noun}"

def outcome(trial):
    """Return the outcome verb ('pousse' or 'tire') for a trial."""
    agent_central = (trial["central_shape"] == trial["agent_shape"])
    same_side = (trial["movement"] == trial["lateral_position"])
    return "pousse" if (agent_central == same_side) else "tire"

def event_description(trial):
    """Return the sentence describing the ground-truth event."""
    return f"{article(trial['agent'])} {trial['outcome']} {article(trial['patient']).lower()}"

def test_sentence(trial):
    """Return the test sentence, possibly with a change from ground-truth."""
    agent, patient, verb, change = trial['agent'], trial['patient'], trial['outcome'], trial.get('change')
    if change:
        swap = SWAP_MAP["verb"] if change == "verb" else SWAP_MAP["nouns"]
        if change == "agent":
            agent = swap.get(agent, agent)
        elif change == "patient":
            patient = swap.get(patient, patient)
        elif change == "verb":
            verb = swap.get(verb, verb)
    return f"{article(agent)} {verb} {article(patient).lower()}"

def check_repetitions(lst, max_consecutive):
    """True iff no element appears more than max_consecutive times in a row."""
    return all(len(list(groups)) <= max_consecutive for _, groups in groupby(lst))

def check_spacing(lst, max_spacing):
    """True iff any two identical elements are at most max_spacing apart."""
    repeat_indices = [i for i, (a, b) in enumerate(pairwise(lst), 1) if a == b]
    return all(j - i <= max_spacing for i, j in pairwise(repeat_indices))

def randomize(lst1, lst2, max_consecutive=1e10, max_spacing=1e10):
    """Return random permutation with repetition and spacing constraints."""
    for _ in range(100000):
        a, b = lst1[:], lst2[:]
        random.shuffle(a)
        random.shuffle(b)
        combined = [x for pair in zip(a, b) for x in pair]
        if check_repetitions(combined, max_consecutive) and check_spacing(combined, max_spacing):
            return combined

def cb_localizer(subject_id, n_trials, n_blocks, start_block=1):
    """Generate counterbalanced localizer trials."""
    rows = []
    for block_num in range(start_block, n_blocks + start_block):
        stims = list(STIMS) * (n_trials // len(STIMS))
        mid = len(stims) // 2
        meanings = randomize(stims[:mid], stims[mid:], max_consecutive=2, max_spacing=30)
        trial_type = ['image' if i % 2 else 'word' for i in range(n_trials)]
        ground_truth = [1 if meanings[i] == meanings[i-1] else 0 for i in range(n_trials)]
        ground_truth[0] = 0
        
        for trial_idx in range(n_trials):
            rows.append({
                "trial_type": trial_type[trial_idx],
                "subject_id": subject_id,
                "block_number": block_num,
                "trial_number": trial_idx + 1,
                "shape1": meanings[trial_idx] if trial_type[trial_idx] == "image" else None,
                "label1": meanings[trial_idx] if trial_type[trial_idx] == "word" else None,
                "ground_truth": ground_truth[trial_idx],
                "correct_key": 'right' if ground_truth[trial_idx] else None,
            })
    return rows

def cb_base(phase, subject_id, shapes, animals, tools, double=True):
    """Generate base trial structure for a phase."""
    shape_pairs = [p for c in combinations(shapes, 2) for p in permutations(c, 2)]
    trials = [
        {"subject_id": subject_id, "shape_pair": sort_key(shape1 + "-" + shape2), "trial_type": phase,
         "shape1": shape1, "shape2": shape2, "agent": animal, "patient": tool,
         "label1": animal if agent == 'shape1' else tool, "label2": tool if agent == 'shape1' else animal,
         "agent_shape": agent, "patient_shape": 'shape2' if agent == 'shape1' else 'shape1', "assignment_order": order}
        for (shape1, shape2), animal, tool, agent, order in 
        product(shape_pairs, animals, tools, ["shape1", "shape2"], ["symbol_first", "referent_first"])
    ]
    
    sort_trials(trials)
    extra_keys = list(MAPPINGS.keys())
    n_groups, mirrored = (2, False) if phase == "training" else (6, True)
    extra_values = [trial for block in balance_vars(len(extra_keys), n_groups, mirrored) for trial in block]
    
    for trial, extra in zip(trials, extra_values):
        trial.update(dict(zip(extra_keys, extra)))
    
    if double:
        trials += [mirror(trial) for trial in trials]
    
    map_AB(trials, MAPPINGS)
    change_order = [key for key, prop in {"agent": 1/4, "patient": 1/4, "verb": 1/2}.items() 
                    for _ in range(int(len(trials) // 2 * prop))]
    random.shuffle(change_order)
    return trials, change_order

def cb_training(subject_id, start_block=0):
    """Generate counterbalanced training trials in 3 phases."""
    trials, change_order = cb_base("training", subject_id, SHAPES_TRAINING, 
                                    ANIMALS_TRAINING, TOOLS_TRAINING, double=False)
    random.shuffle(trials)
    block_1_end = TRAINING_BLOCK_1_SIZE
    block_2_end = block_1_end + TRAINING_BLOCK_2_SIZE
    
    for trial_idx, trial in enumerate(trials, 1):
        if trial_idx <= block_1_end:
            block_info = (0, "training_no_assignment", trial_idx)
        elif trial_idx <= block_2_end:
            block_info = (1, "training_assignment", trial_idx - block_1_end)
        else:
            block_info = (2, "training_no_animation", trial_idx - block_2_end)
        
        block_idx, trial_type, trial_num = block_info
        trial.update({"block_number": block_idx, "trial_type": trial_type, "trial_number": trial_num})
        
        if block_idx == 0:
            trial.update({
                "label1": trial["shape1"], "label2": trial["shape2"],
                "agent": trial["shape1"] if trial["agent_shape"] == "shape1" else trial["shape2"],
                "patient": trial["shape2"] if trial["agent_shape"] == "shape1" else trial["shape1"]
            })
        
        trial["outcome"] = outcome(trial)
        trial["change"] = change_order.pop(0) if trial.get("correct_key") == "left" else None
        trial["ground_truth"] = event_description(trial)
        trial["test_sentence"] = test_sentence(trial)
    
    return trials

def cb_main(subject_id, start_block=1, double=True):
    """Generate counterbalanced main experiment trials."""
    for _ in range(1000):
        trials, change_order = cb_base("test", subject_id, SHAPES, ANIMALS, TOOLS, double=double)
        
        for trial in trials:
            trial["outcome"] = outcome(trial)
            trial["change"] = change_order.pop(0) if trial.get("correct_key") == "left" else None
            trial["ground_truth"] = event_description(trial)
            trial["test_sentence"] = test_sentence(trial)
        
        if is_balanced(trials):
            break
    
    random.shuffle(trials)
    add_numbers(trials, start_from=start_block)
    return trials

if __name__ == '__main__':
    cb = []
    for subject_id in range(1, N_SUBJ + 1):
        print(f"Generating counterbalancing for subject {subject_id}...")
        cb.extend(cb_localizer(subject_id, n_trials=64, n_blocks=1, start_block=0))
        cb.extend(cb_localizer(subject_id, n_trials=480, n_blocks=2, start_block=1))
        cb.extend(cb_training(subject_id))
        cb.extend(cb_main(subject_id))
    
    with open(COUNTERBALANCE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLNAMES)
        writer.writeheader()
        writer.writerows(cb)