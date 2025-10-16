"""
Constants for the Pushmi MEG Experiment
========================================

This module contains all configuration constants for the experiment,
organized by category for easy maintenance and reference.
"""
from pathlib import Path

# =============================================================================
# EXPERIMENT DESIGN PARAMETERS
# =============================================================================
N_SUBJ = 30          # Total number of subjects
N_BLOCKS = 12        # Number of experimental blocks
N_TRIALS = 384       # Total number of trials
BLOCK_SIZE = 32      # Trials per block

# Training block sizes
TRAINING_BLOCK_1_SIZE = 6   # Phase 1: No assignment (shapes only)
TRAINING_BLOCK_2_SIZE = 10  # Phase 2: With assignment
TRAINING_BLOCK_3_SIZE = 20  # Phase 3: No animation

# =============================================================================
# DISPLAY & VISUAL PARAMETERS
# =============================================================================

SCALE_FACTOR = 2

# Colors (RGB tuples)
GREEN = (153, 229, 153)      # Correct feedback
RED = (240, 128, 128)        # Incorrect feedback
LIGHTGRAY = (211, 211, 211)  # Timeout feedback

# Stimulus dimensions (scaled)
TEXTSIZE = int(24 * SCALE_FACTOR)
SHAPE_WIDTH = int(40 * SCALE_FACTOR)
IMAGE_WIDTH = int(60 * SCALE_FACTOR)
HALFWIDTH = SHAPE_WIDTH // 2
DISPLACEMENT = int(100 * SCALE_FACTOR)

# =============================================================================
# TIMING PARAMETERS (all in milliseconds)
# =============================================================================

# Main experiment trial timing
INITIAL_T = 500           # Initial fixation
ASSIGNMENT_T = 1000       # Shape-label assignment display
POST_ASSIGN_T = 1000      # Post-assignment fixation
CENTRAL_FLASH_T = 1000    # Central shape flash
POST_FLASH_T = 800        # Post-flash fixation
LATERAL_FLASH_T = 200     # Lateral shape flash
OUTCOME_T = 200           # Outcome display (push/pull)
POST_OUTCOME_T = 500      # Post-outcome fixation
WORD_T = 400              # Word presentation in test sentence
RESPONSE_T = 1800         # Response window
FEEDBACK_T = 200          # Feedback display

# Eyetracker calibration timing (currently unused)
TEXT_DISPLAY_TIME = 5000
BLANK_DISPLAY_TIME = 2000
CIRCLE_DISPLAY_TIME = 1500
CIRCLE_RADIUS = 20
TEXT_SIZE = 50

# =============================================================================
# STIMULUS SETS
# =============================================================================
# Main experiment stimuli
ANIMALS = ('louve', 'poule')                    # Animals (feminine)
TOOLS = ('tasse', 'malle')                      # Tools (feminine)
SHAPES = ('carre', 'cercle', 'etoile', 'croix') # Shapes (masculine)

# Training stimuli (separate set to avoid interference)
SHAPES_TRAINING = ('pentagone', 'losange')
ANIMALS_TRAINING = ('mouton', 'pigeon')
TOOLS_TRAINING = ('marteau', 'couteau')

# Combined sets
STIMS = (*ANIMALS, *TOOLS, *SHAPES)
ALL_SHAPES = SHAPES + SHAPES_TRAINING

# Stimulus sizes (scale factors)
SIZES = {
    k: (IMAGE_WIDTH if k in ['louve', 'malle'] else SHAPE_WIDTH) / 1000 
    for k in STIMS + SHAPES_TRAINING
}

# =============================================================================
# FRENCH LANGUAGE PROPERTIES
# =============================================================================
# Gender classification for article selection
MASC_NAMES = ('mouton', 'pigeon', 'marteau', 'couteau', 'pentagone', 'losange')
FEM_NAMES = ('louve', 'poule', 'tasse', 'malle')

# Accented characters mapping
ACCENTS = {
    "carre": "carré", 
    "etoile": "étoile"
}

# Noun pairs for change trials (swapping referents)
NOUN_PAIRS = [
    ("louve", "poule"),           # Animals
    ("tasse", "malle"),           # Tools
    ("mouton", "pigeon"),         # Training animals
    ("marteau", "couteau"),       # Training tools
    ("losange", "ellipse"),       # Training shapes
    ("pentagone", "hexagone"),    # Training shapes
]

# Bidirectional swap mappings for change trials
SWAP_MAP = {
    "nouns": {a: b for a, b in NOUN_PAIRS} | {b: a for a, b in NOUN_PAIRS},
    "verb": {"pousse": "tire", "tire": "pousse"},
}

# All possible words in test sentences
SENTENCE_STIMS = (
    SHAPES + 
    MASC_NAMES + 
    FEM_NAMES + 
    ('La', 'la', 'Le', 'le', 'pousse', 'tire', 
     "L'ellipse", "l'ellipse", "L'hexagone", "l'hexagone")
)


# =============================================================================
# RESPONSE MAPPING
# =============================================================================
# Map keyboard keys and MEG buttons to response labels
KEYMAP = {
    ord('f'): 'left',   # Left keyboard key
    ord('j'): 'right',  # Right keyboard key
    'LR': 'left',       # Left MEG button (red)
    'RR': 'right',      # Right MEG button (red)
}

# =============================================================================
# COUNTERBALANCING
# =============================================================================
# Column names for counterbalancing CSV
COLNAMES = [
    "subject_id", "trial_type", "block_number", "trial_number", 
    "assignment_order", "shape_pair", "shape1", "label1", "shape2", "label2", 
    "agent", "patient", "agent_shape", "patient_shape", 
    "central_shape", "lateral_position", "movement", "outcome", 
    "ground_truth", "change", "test_sentence", "correct_key"    
]

# A/B factor mappings for counterbalancing
MAPPINGS = {
    'central_shape': {'A': 'shape1', 'B': 'shape2'},
    'lateral_position': {'A': 'left', 'B': 'right'},
    'movement': {'A': 'left', 'B': 'right'},
    'correct_key': {'A': 'right', 'B': 'left'}
}

# =============================================================================
# FILE PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
STIM_DIR = BASE_DIR.parent / "Stimuli"
COUNTERBALANCE_CSV = BASE_DIR.parent / "Randomization" / "cb.csv"

# =============================================================================
# MEG HARDWARE CONFIGURATION
# =============================================================================
# Map raw parallel port status codes to button labels
# Each port has different buttons with unique status codes
BUTTONS = {
    'port1': {1: 'RB', 2: 'LY', 4: 'LR', 8: 'LG'},  # Red/Blue, Left Yellow/Red/Green
    'port2': {2: 'RR', 4: 'RG', 8: 'RY'},            # Right Red/Green/Yellow
    'port3': {8: 'LB'}                               # Left Blue
}

# Map event names to integer trigger codes sent to MEG
TRIGGERS = {
    # MEG baseline: set_data(0)
    'baseline': 0,
    
    # Assignment phase
    'fix_init': 1,
    'assign_1_shape': 2,
    'assign_1_label': 3,
    'fix_inter_assign': 4,
    'assign_2_shape': 5,
    'assign_2_label': 6,
    'fix_post_assign': 7,
    
    # Animation/flash phase
    'central_flash': 8,
    'fix_inter_flash': 9,
    'lateral_flash': 10,
    'fix_post_flash': 11,
    
    # Outcome phase
    'outcome': 12,
    'fix_post_outcome': 13,
    
    # Test sentence (5 words)
    'test_1': 14,
    'test_2': 15,
    'test_3': 16,
    'test_4': 17,
    'test_5': 18,
    
    # Response and feedback
    'response': 19,
    'fb_correct': 20,
    'fb_incorrect': 21,
    'fb_timeout': 22,
    
    # Localizer
    'word': 23,
    'image': 24,
}

# =============================================================================
# INSTRUCTION SCREEN INDICES
# =============================================================================
INSTRUCTIONS = {
    'localizer': 1,
    'training_intro': 2,
    'training_assignment': 3,
    'training_no_animation': 4,
    'training_animation': 5,
    'main_experiment': 6,
}