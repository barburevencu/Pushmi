from os import name
import random, csv, copy
from constants import *
from itertools import groupby, pairwise, permutations, product, combinations

"""
Constants for Pushmi MEG counterbalancing and sentence generation.

Stimulus sets
-------------
ANIMALS: Animal labels used in sentences/trials.
TOOLS: Object labels used in sentences/trials.
SHAPES: Visual shapes used to form ordered/ unordered pairs in trials.
STIMS: Convenience union of ANIMALS + TOOLS + SHAPES.

Grammatical gender & swapping
-------------------------
MASC_NAMES: Nouns that take the masculine article "Le".
FEM_NAMES: Nouns that take the feminine article "La".
NOUN_PAIRS: Bidirectional noun swaps (A↔B) used to generate "changed" test sentences.
SWAP_MAP: Lookup tables for one-step swaps
    - "nouns": bidirectional swaps built from NOUN_PAIRS
    - "verb" : {'pousse'↔'tire'} to flip the predicate in test sentences.

Sentence vocabulary
-------------------
SENTENCE_STIMS: Tokens that may appear in rendered sentences (shapes, nouns, articles, verbs).

Key mapping
-------------------
KEYMAP: Maps device/readout keys to logical responses
    ord('f') → 'left', ord('j') → 'right', 'LR' → 'left', 'RR' → 'right'.

Counterbalancing I/O schema
---------------------------
COLNAMES:
    Column order for exported trial CSVs:
      [
        "subject_id", "trial_type", "block_number", "trial_number",
        "assignment_order", "shape_pair", "shape1", "label1", "shape2", "label2",
        "agent", "patient", "agent_shape", "patient_shape",
        "central_shape", "lateral_position", "movement", "outcome",
        "ground_truth", "change", "test_sentence", "correct_key"
      ]

A/B → concrete value mappings
-----------------------------
MAPPINGS: Field-wise mapping from abstract A/B codes to concrete values
    'central_shape'   : {'A':'shape1', 'B':'shape2'}
    'lateral_position': {'A':'left',   'B':'right'}
    'movement'        : {'A':'left',   'B':'right'}
    'correct_key'     : {'A':'right',  'B':'left'}
"""

def shuffled_copies(lst, n):
    """Return n unique shuffled versions of lst (up to all possible permutations)."""
    copies, seen = [], set()
    while len(copies) < n:
        new_list = lst[:]          # shallow copy; we don't mutate inner rows
        random.shuffle(new_list)
        sig = tuple(id(x) for x in new_list)  # signature by object order
        if sig not in seen:
            seen.add(sig)
            copies.append(new_list)
    return copies

def mirror(obj, elem1='A', elem2='B'):
    """Swap elem1<->elem2 anywhere inside arbitrarily nested dict/list/tuple."""
    if isinstance(obj, dict):
        return {k: mirror(v, elem1, elem2) for k, v in obj.items()}
    if isinstance(obj, list):
        return [mirror(x, elem1, elem2) for x in obj]
    if isinstance(obj, tuple):
        return tuple(mirror(x, elem1, elem2) for x in obj)
    return elem2 if obj == elem1 else elem1 if obj == elem2 else obj

def balance_vars(n_vars=4, n_groups=6, mirrored=True):
    """Return a list of n_groups * 2 lists, each with a balanced combination of n_vars A/B factors."""
    full_factor = [list(prod) for prod in product(['A', 'B'], repeat=n_vars)]
    shuffled = shuffled_copies(full_factor, n_groups)
    if not mirrored:
        return shuffled
    shuffled_mirror = mirror(shuffled, "A", "B")
    combined = shuffled + shuffled_mirror
    
    for _ in range(10000):
        random.shuffle(combined)
        valid_shuffle = all(combined[i] != mirror(combined[i + 1]) for i in range(0, len(combined), 2))
        if valid_shuffle:
            return combined
        
    raise ValueError("Couldn't find a valid shuffling after 10000 attempts.")
        
def is_balanced(trials, cols=('assignment_order', 'agent', 'patient', 'correct_key', 'central_shape')):
    """Return True if each level of each column has equal 'pousse' and 'tire' outcomes."""
    for col in cols:
        # sort so groupby works
        sorted_trials = sorted(trials, key=lambda t: t[col])
        for _, group in groupby(sorted_trials, key=lambda t: t[col]):
            pousse = tire = 0
            for t in group:
                pousse += (t['outcome'] == 'pousse')
                tire   += (t['outcome'] == 'tire')
            if pousse != tire:
                return False
    return True

def map_AB(trials, mappings):
    """Map A/B fields to concrete values in place."""
    for t in trials:
        for col, mapping in mappings.items():
            if col in t:
                v = t[col]
                t[col] = mapping.get(v, v)

def add_numbers(trials, start_from=1):
    """Shuffle trials and assign block_number and trial_number."""
    for i, trial in enumerate(trials):
        trial["block_number"] = (i // BLOCK_SIZE) + start_from
        trial["trial_number"] = (i % BLOCK_SIZE) + 1

def sort_key(s):
    """Return a consistent key for a shape pair 'A-B' or 'B-A'."""
    a, b = s.split("-")
    return "-".join(sorted((a, b)))

def sort_trials(trials):
    """Sort trials by shape_pair, agent_shape, agent, patient, shape1-shape2, assignment_order."""
    trials.sort(key=lambda d: (
          d['shape_pair'], 
          d["agent_shape"], 
          d["agent"], d["patient"], 
          d['shape1'] + "-" + d['shape2'], 
          d["assignment_order"]
        ))

def starts_with_vowel(word):
    """Return True if a word starts with a vowel (a, e, i, o, u, y, é, é)."""
    return word[0].lower() in "aeiouhyééè"

def article(noun):
    """Return the correct French definite article for a noun."""
    if starts_with_vowel(noun):
        return f"L'{noun}"
    if noun in MASC_NAMES:
        return f"Le {noun}"
    if noun in FEM_NAMES:
        return f"La {noun}"
    return f"Le {noun}"

def outcome(trial):
    """Return the outcome verb ('pousse' or 'tire') for a trial."""
    agent_central = (trial["central_shape"] == trial["agent_shape"])
    same_side  = (trial["movement"] == trial["lateral_position"])
    return "pousse" if (agent_central == same_side) else "tire"

def event_description(trial):
    """Return the sentence describing the ground-truth event."""
    agent, patient, verb = (trial[k] for k in ('agent', 'patient', 'outcome'))
    return f"{article(agent)} {verb} {article(patient).lower()}"

def test_sentence(trial):
    """Return the test sentence, possibly with a change from the ground-truth event."""
    agent, patient, verb, change = (trial[k] for k in ('agent', 'patient', 'outcome', 'change'))
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

def randomize(lst, max_consecutive=1e10, max_spacing=1e10, max_attempts=100000):
    """Return a random permutation of lst with constraints on repetitions and spacing."""
    cb = lst[:]
    for _ in range(max_attempts):
        random.shuffle(cb)
        if check_repetitions(cb, max_consecutive) and check_spacing(cb, max_spacing):
            return cb
    raise RuntimeError("randomize() could not satisfy constraints in max_attempts")

def cb_localizer(subject_id, n_trials, n_blocks, start_block=1):
    """Generate a counterbalanced list of localizer trials."""
    n_repeats = n_trials // len(STIMS)
    rows = []

    for b in range(start_block, n_blocks + start_block):
        meanings = randomize(list(STIMS) * n_repeats, max_consecutive=2, max_spacing=30)
        trial_type = ['image' if i % 2 else 'word' for i in range(n_trials)]
        ground_truth = [1 if meanings[i] == meanings[i-1] else 0 for i in range(n_trials)]
        ground_truth[0] = 0
        correct_key = ['right' if gt else None for gt in ground_truth]

        for t in range(n_trials):
            rows.append({
                "trial_type": trial_type[t],
                "subject_id": subject_id,
                "block_number": b,
                "trial_number": t + 1,
                "shape1": meanings[t] if trial_type[t] == "image" else None,
                "label1": meanings[t] if trial_type[t] == "word" else None,
                "ground_truth": ground_truth[t],
                "correct_key": correct_key[t],
            })

    return rows

def cb_base(phase, subject_id, shapes, animals, tools, double=True):
    shape_pairs = [p for c in combinations(shapes, 2) for p in permutations(c, 2)]
    agent_shapes = ["shape1", "shape2"]
    assignment_order = ["symbol_first", "referent_first"]
    trials = [
                {
                    "subject_id": subject_id,
                    "shape_pair": sort_key(shape1 + "-" + shape2),
                    "trial_type": phase,
                    "shape1": shape1, 
                    "shape2": shape2,
                    "agent": animal,
                    "patient": tool,
                    "label1": animal if agent == 'shape1' else tool,
                    "label2": tool if agent == 'shape1' else animal,
                    "agent_shape": agent,
                    "patient_shape": 'shape2' if agent == 'shape1' else 'shape1',
                    "assignment_order": order
                }
                for (shape1, shape2), animal, tool, agent, order in 
                product(shape_pairs, animals, tools, agent_shapes, assignment_order)
            ]
            
    sort_trials(trials)

    extra_keys = list(MAPPINGS.keys())
    n_groups, mirrored = (2, False) if phase == "training" else (6, True)
    extra_values = [trial for block in balance_vars(n_vars=len(extra_keys), n_groups=n_groups, mirrored=mirrored) for trial in block]

    for trial, extra in zip(trials, extra_values):
        trial.update(dict(zip(extra_keys, extra)))

    if double: 
        trials += [mirror(t) for t in trials]

    map_AB(trials, MAPPINGS)
    num_false = len(trials) // 2
    split = {"agent": 1/4, "patient": 1/4, "verb": 1/2}
    
    change_order = [k for k, p in split.items() for _ in range(int(num_false * p))]
    random.shuffle(change_order)
    change_order = iter(change_order)

    return trials, change_order

def cb_training(subject_id, start_block=0):
    """Generate a counterbalanced list of training trials."""
    trials, change_order = cb_base("training", subject_id, SHAPES_TRAINING, ANIMALS_TRAINING, TOOLS_TRAINING, double=False)

    random.shuffle(trials)
    add_numbers(trials, start_from=start_block)

    for ti, t in enumerate(trials, 1):
        bi = 0 if ti <= 6 else 1 if ti <= 16 else 2
        t["label1"] = t["shape1"] if bi == 0 else t["label1"]
        t["label2"] = t["shape2"] if bi == 0 else t["label2"]
        t["block_number"] = bi
        t["trial_type"] = "training_no_assignment" if bi == 0 else "training_assignment" if bi == 0 else "training_no_animation"
        t["trial_number"] = ti if bi == 0 else ti - 6 if bi == 1 else ti - 16
        t["outcome"] = outcome(t)
        t["change"] = next(change_order, None) if t.get("correct_key") == "left" else None
        t["agent"] = (t["shape1"] if t["agent_shape"] == "shape1" else t["shape2"]) if bi == 0 else t["agent"]
        t["patient"]= (t["shape1"] if t["agent_shape"] == "shape2" else t["shape2"]) if bi == 0 else t["patient"]
        t["ground_truth"] = event_description(t)
        t["test_sentence"] = test_sentence(t)

    return trials

def cb_main(subject_id, start_block=1, double=True):
    """Generate a counterbalanced list of main experiment trials."""
    balanced = False
    while not balanced:
        trials, change_order = cb_base("test", subject_id, SHAPES, ANIMALS, TOOLS, double=double)

        for t in trials:
            t["outcome"] = outcome(t)
            t["change"] = next(change_order, None) if t.get("correct_key") == "left" else None
            t["ground_truth"] = event_description(t)
            t["test_sentence"] = test_sentence(t)

        if is_balanced(trials, cols=['assignment_order', 'agent', 'patient', 'correct_key', 'central_shape']):
            balanced = True

    random.shuffle(trials)
    add_numbers(trials, start_from=start_block)

    return trials

if __name__ == '__main__':
    cb = []
    for subject_id in range(1, 31):
        cb.extend(cb_localizer(subject_id, n_trials=64, n_blocks=1, start_block=0))
        cb.extend(cb_localizer(subject_id, n_trials=480, n_blocks=2, start_block=1))
        cb.extend(cb_training(subject_id))
        cb.extend(cb_main(subject_id, double=True))

    with open(COUNTERBALANCE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLNAMES)
        writer.writeheader()
        writer.writerows(cb)