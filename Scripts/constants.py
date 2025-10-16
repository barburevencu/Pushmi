from pathlib import Path

N_SUBJ = 30

# Training block size constants
TRAINING_BLOCK_1_SIZE = 6   # no assignment
TRAINING_BLOCK_2_SIZE = 10  # with assignment
TRAINING_BLOCK_3_SIZE = 20  # no animation


SCALE_FACTOR = 1.5

GREEN = (153, 229, 153)
RED = (240, 128, 128)
LIGHTGRAY = (211, 211, 211)

OFFSET_X = 0 #-330

TEXTSIZE = int(24 * SCALE_FACTOR)
SHAPE_WIDTH = int(40 * SCALE_FACTOR)
IMAGE_WIDTH = int(60 * SCALE_FACTOR)
HALFWIDTH = SHAPE_WIDTH // 2
DISPLACEMENT = int(100 * SCALE_FACTOR)

ORIGIN = (0 + OFFSET_X, 0)

# Define eyetracker constants
TEXT_DISPLAY_TIME = 5000
BLANK_DISPLAY_TIME = 2000
CIRCLE_DISPLAY_TIME = 1500
CIRCLE_RADIUS = 20
TEXT_SIZE = 50

# Define trial timing constants
INITIAL_T = 500
ASSIGNMENT_T = 1000
POST_ASSIGN_T = 1000
CENTRAL_FLASH_T = 1000
POST_FLASH_T = 800
LATERAL_FLASH_T = 200
OUTCOME_T = 200
POST_OUTCOME_T = 500
WORD_T = 400
RESPONSE_T = 1800
FEEDBACK_T = 200

# Stimuli
ANIMALS = ('louve', 'poule')
TOOLS = ('tasse', 'malle')
SHAPES = ('carre', 'cercle', 'etoile', 'croix')

SHAPES_TRAINING = ('pentagone', 'losange')
ANIMALS_TRAINING = ('mouton', 'pigeon')
TOOLS_TRAINING = ('marteau', 'couteau')

STIMS = (*ANIMALS, *TOOLS, *SHAPES)
SIZES = {k: (IMAGE_WIDTH if k in ['louve', 'malle'] else SHAPE_WIDTH) / 1000 for k in STIMS + SHAPES_TRAINING}

MASC_NAMES = ('mouton', 'pigeon', 'marteau', 'couteau', 'pentagone', 'losange')
FEM_NAMES = ('louve', 'poule', 'tasse', 'malle')

NOUN_PAIRS = [
    ("louve", "poule"), ("tasse", "malle"), 
    ("mouton", "pigeon"), ("marteau", "couteau"), 
    ("losange", "ellipse"), ("pentagone", "hexagone"),
]

ACCENTS = {"carre": "carré", "etoile": "étoile"}

SWAP_MAP = {
    "nouns": {a: b for a, b in NOUN_PAIRS} | {b: a for a, b in NOUN_PAIRS},
    "verb": {"pousse": "tire", "tire": "pousse"},
}

SENTENCE_STIMS = SHAPES + MASC_NAMES + FEM_NAMES + ('La', 'la', 'Le', 'le', 'pousse', 'tire', "L'ellipse", "l'ellipse", "L'hexagone", "l'hexagone")

KEYMAP = {k: v for k, v in zip((ord('f'), ord('j'), 'LR', 'RR'), ('left', 'right', 'left', 'right'))}

# Counterbalancing and Paths
COLNAMES = [
        "subject_id", "trial_type", "block_number", "trial_number", 
        "assignment_order", "shape_pair", "shape1", "label1", "shape2", "label2", 
        "agent", "patient", "agent_shape", "patient_shape", 
        "central_shape", "lateral_position", "movement", "outcome", 
        "ground_truth", "change", "test_sentence", "correct_key"    
    ]

MAPPINGS = {
    'central_shape': {'A': 'shape1', 'B': 'shape2'},
    'lateral_position': {'A': 'left', 'B': 'right'},
    'movement': {'A': 'left', 'B': 'right'},
    'correct_key': {'A': 'right', 'B': 'left'}
}

BASE_DIR = Path(__file__).resolve().parent
STIM_DIR = BASE_DIR.parent / "Stimuli"
COUNTERBALANCE_CSV = BASE_DIR.parent / "Randomization" / "cb.csv"

# MEG
# Map raw port codes to button labels per port
BUTTONS = {
    'port1': {1: 'RB', 2: 'LY', 4: 'LR', 8: 'LG'},
    'port2': {2: 'RR', 4: 'RG', 8: 'RY'},
    'port3': {8: 'LB'}
    }

# Maps event names to integer trigger codes for MEG
TRIGGERS = {
    'baseline': 0,
    'fix_init': 1,
    'assign_1_shape': 2, 'assign_1_label': 3,
    'fix_inter_assign': 4,
    'assign_2_shape': 5, 'assign_2_label': 6,
    'fix_post_assign': 7,
    'central_flash': 8,
    'fix_inter_flash': 9,
    'lateral_flash': 10,
    'fix_post_flash': 11,
    'outcome': 12,
    'fix_post_outcome': 13,
    'test_1': 14, 'test_2': 15, 'test_3': 16, 'test_4': 17, 'test_5': 18, 
    'response': 19, 'fb_correct': 20, 'fb_incorrect': 21, 'fb_timeout': 22,
    'word': 23, 'image': 24
}

# Trial parameters
N_BLOCKS = 12
N_TRIALS = 384
BLOCK_SIZE = 32
 