import pandas as pd, argparse
from dataclasses import dataclass
from typing import Optional
from expyriment import control, design, stimuli, io
from expyriment.misc.constants import C_BLACK, C_WHITE, C_GREY, K_f, K_j, K_SPACE
from constants import *
from meg import HardwareManager
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger()

@dataclass
class ExperimentConfig:
    """Configuration for experiment hardware and behavior."""
    subject_id: int
    meg: bool = False
    eyetracker: bool = False
    localizer: bool = False
    training: bool = False
    fullscreen: bool = False
    
    @classmethod
    def from_args(cls, args, flags):
        """Create config from parsed arguments and flags."""
        return cls(
            subject_id=args.subject_id,
            meg='meg' in flags,
            eyetracker='eyetracker' in flags,
            localizer='localizer' in flags,
            training='training' in flags,
            fullscreen='fullscreen' in flags
        )
    
    def setup_directories(self):
        """Configure data directories based on fullscreen mode."""
        if self.fullscreen:
            logger.disabled = True
            base_dir = f"../Data/Pilot/Behavior/sub-{self.subject_id:02d}"
            io.defaults.datafile_directory = base_dir
            io.defaults.eventfile_directory = base_dir

""" Helper functions """
def preload(stimulus):
    stimulus.preload()
    return stimulus

def parse_blocks(df, condition):
    return [
        g.to_dict("records")
        for _, g in df.loc[condition].groupby("block_number", sort=True)
    ]

def picture(name, scale_factor):
    stim = stimuli.Picture(str(STIM_DIR / f"{name}.png"), position=ORIGIN)
    stim.scale(scale_factor)
    return preload(stim)

def word(name):
    return preload(stimuli.TextLine(text=ACCENTS.get(name, name), position=ORIGIN, text_size=TEXTSIZE))

def diff(t0):
    return exp.clock.time - t0

def present(*stims, duration=None, keys=None, event=None):
    t0 = exp.clock.time # Start timer
    
    # Draw stimuli
    exp.screen.clear() 
    for s in stims: 
        s.present(False, False)
    fixation.present(False, True)

    # Send trigger if applicable
    if hardware.meg_handler and hardware.port and event:
        hardware.port.set_data(TRIGGERS[event])
    if hardware.eyetracker and event and event != "baseline":
        hardware.eyetracker.send_message(event)

    if not duration and not keys: 
        return None, None

    # Keep on-screen for target duration
    remaining = None if duration is None else max(0, duration - diff(t0))

    if keys:
        kb = hardware.meg_handler if hardware.meg_handler else exp.keyboard
        key, rt = kb.wait(keys, remaining)
        return KEYMAP.get(key), rt
    
    if remaining is not None:
        exp.clock.wait(remaining)
    
    return None, None

def show(message):
    """Display a message and wait for spacebar press."""
    message.present()
    exp.keyboard.wait(K_SPACE)

def draw(*stims, duration=None, keys=None, event=None):
    t0 = exp.clock.time
    present(*stims, pulse, duration=50, event=event)
    key, rt = present(*stims, duration=duration-50, keys=keys, event='baseline')
    logger.info(f"{event} duration: {exp.clock.time - t0}")
    return key, rt

def draw_seq(steps):
    for stims, dur, evt in steps:
        if not stims:
            draw(duration=dur, event=evt)
        elif isinstance(stims, (list, tuple)):
            draw(*stims, duration=dur, event=evt)
        else:
            draw(stims, duration=dur, event=evt)

def run_localizer_trial(**params): 
    """Run a single localizer trial with error handling."""
    t0 = exp.clock.time
    
    try:
        trial_type, shape1, label1, correct_key = (params.get(k) for k in ('trial_type', 'shape1', 'label1', 'correct_key'))
        
        # Validate stimulus exists
        if shape1 and shape1 not in images:
            raise KeyError(f"Image stimulus not found: {shape1}")
        if label1 and label1 not in words:
            raise KeyError(f"Word stimulus not found: {label1}")
        
        stim = images[shape1] if shape1 else words[label1]

        key, rt = draw(stim, duration=800, keys=hardware.response_keys, event=trial_type)
        correct = key == correct_key
        
        if not correct:
            present(feedback['incorrect'], stim, duration=100)
            present(stim)

        params.update(timestamp=t0, response=key, rt=rt, correct=correct)
        exp.data.add([params.get(k) for k in var_names])
        exp.clock.wait(1000 - diff(t0))
        
    except (KeyError, ValueError) as e:
        print(f"Error in localizer trial: {e}")
        params.update(timestamp=t0, response=None, rt=None, correct=False)
        exp.data.add([params.get(k) for k in var_names])

def final_positions(movement, lateral_position):
    """Return (central_x, lateral_x) given movement direction and lateral side."""
    right = (DISPLACEMENT + HALFWIDTH + OFFSET_X,  DISPLACEMENT - HALFWIDTH + OFFSET_X)
    left  = (-(DISPLACEMENT - HALFWIDTH) + OFFSET_X, -(DISPLACEMENT + HALFWIDTH) + OFFSET_X)
    pos = right if movement == "right" else left
    return pos if lateral_position == "left" else pos[::-1]

def assignment_seq(order, s1, l1, s2, l2):
    """Build the assignment sequence based on whether shape or label comes first."""
    a1 = (s1, ASSIGNMENT_T, "assign_1_shape") if order == "shape_first" else (l1, ASSIGNMENT_T, "assign_1_label")
    a2 = (l1, ASSIGNMENT_T, "assign_1_label") if order == "shape_first" else (s1, ASSIGNMENT_T, "assign_1_shape")
    a3 = (s2, ASSIGNMENT_T, "assign_2_shape") if order == "shape_first" else (l2, ASSIGNMENT_T, "assign_2_label")
    a4 = (l2, ASSIGNMENT_T, "assign_2_label") if order == "shape_first" else (s2, ASSIGNMENT_T, "assign_2_shape")
    return [(None, INITIAL_T, "fix_init"), a1, a2, (None, POST_ASSIGN_T, "fix_inter_assign"), a3, a4, (None, POST_ASSIGN_T, "fix_post_assign")]

def move_pair_to_x(central, lateral, target_x, speed=3):
    """Slide both shapes horizontally until target_x reached."""
    current_x = central.position[0]
    step = speed if target_x > current_x else -speed
    while (target_x - current_x) * step > 0:
        central.move((step, 0))
        lateral.move((step, 0))
        current_x = central.position[0]
        if target_x != current_x:
            present(central, lateral)

def run_main_trial(**params):
    """Run a single main experiment trial with error handling."""
    t0 = exp.clock.time
    
    try:
        # Validate required parameters
        required_keys = ['shape1', 'label1', 'shape2', 'label2', 'trial_type',
                        'test_sentence', 'correct_key', 'assignment_order',
                        'central_shape', 'lateral_position', 'movement']
        
        missing = [key for key in required_keys if key not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        
        # Stimuli
        shape1_id, label1_id, shape2_id, label2_id = (params[k] for k in ('shape1', 'label1', 'shape2', 'label2'))
        
        # Validate stimuli exist
        if shape1_id not in images or shape2_id not in images:
            raise KeyError(f"Shape stimulus not found: {shape1_id} or {shape2_id}")
        if label1_id not in words or label2_id not in words:
            raise KeyError(f"Word stimulus not found: {label1_id} or {label2_id}")
        
        shape1, shape2 = (images[s] for s in (shape1_id, shape2_id))
        label1, label2   = (words[w] for w in (label1_id, label2_id))

        # Trial parameters
        trial_type, test_sentence, correct_key, order = (params[k] for k in ('trial_type', 'test_sentence', 'correct_key', 'assignment_order'))
        central_key, lateral_position, movement = (params[k] for k in ('central_shape', 'lateral_position', 'movement'))
        central_shape = shape1 if central_key == 'shape1' else shape2
        lateral_shape = shape2 if central_key == 'shape1' else shape1
        
        # Positions
        offset = -SHAPE_WIDTH if lateral_position == 'left' else SHAPE_WIDTH
        central_x, lateral_x = final_positions(movement, lateral_position)

        # ASSIGNMENTS
        draw_seq(assignment_seq(order, shape1, label1, shape2, label2))

        # Prepare lateral flash position
        lateral_shape.move((offset, 0))

        if trial_type in ['test', 'training_no_animation']:
            draw_seq([
                (central_shape, CENTRAL_FLASH_T,  'central_flash'),
                (None,          POST_FLASH_T,     'fix_inter_flash'),
                (lateral_shape, LATERAL_FLASH_T,  'lateral_flash'),
                (None,          POST_FLASH_T,     'fix_post_flash')
            ])

        else:
            draw_seq([
                (central_shape,                  CENTRAL_FLASH_T, 'central_flash'),
                ([central_shape, lateral_shape], CENTRAL_FLASH_T, 'central_flash'),
            ])

        # MOVE SHAPES
        if trial_type not in ['test', 'training_no_animation']:
            move_pair_to_x(central_shape, lateral_shape, central_x)

        central_shape.reposition((central_x, 0))
        lateral_shape.reposition((lateral_x, 0))
        
        # OUTCOME (200 ms)
        draw_seq([
            ((central_shape, lateral_shape), OUTCOME_T,      'outcome'),
            (None,                           POST_OUTCOME_T, 'fix_post_outcome')
        ])

        # TEST SENTENCE (400 ms × 5 = 2000 ms)
        for i, w in enumerate(test_sentence.split(' '), 1):
            draw(words[w], duration=WORD_T, event=f'test_{i}')

        # RESPONSE (max 1800 ms)
        key, rt = draw(duration=RESPONSE_T, keys=hardware.response_keys, event='response')
        correct = key == correct_key

        shape1.reposition(ORIGIN)
        shape2.reposition(ORIGIN)

        fb_key = "timeout" if rt is None else ("correct" if correct else "incorrect")
        
        params.update(timestamp=t0, response=key, rt=rt, correct=correct)
        exp.data.add([params.get(k) for k in var_names])

        draw(feedback[fb_key], duration=FEEDBACK_T, event=f"fb_{fb_key}")
        
    except (KeyError, ValueError) as e:
        print(f"Error in main trial: {e}")
        params.update(timestamp=t0, response=None, rt=None, correct=False)
        exp.data.add([params.get(k) for k in var_names])
    except Exception as e:
        print(f"Unexpected error in main trial: {e}")
        params.update(timestamp=t0, response=None, rt=None, correct=False)
        exp.data.add([params.get(k) for k in var_names])
        raise  # Re-raise unexpected errors

def take_break(message):
    if hardware.eyetracker: 
        hardware.eyetracker.stop_recording()
    show(message)
    if hardware.eyetracker: 
        hardware.eyetracker.start_new_block()
        hardware.eyetracker.start_recording()

if __name__ == '__main__':
    """ GLOBAL SETTINGS """
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=int, default=2)
    args, extras = parser.parse_known_args()
    
    # Create configuration
    config = ExperimentConfig.from_args(args, set(extras))
    
    # Setup environment
    if not config.fullscreen:
        control.set_develop_mode(skip_wait_methods=False)
    else:
        config.setup_directories()

    """ HARDWARE SETUP """
    hardware = HardwareManager(
        subject_id=config.subject_id,
        meg=config.meg,
        eyetracker=config.eyetracker
    ).setup()

    """ DESIGN """
    cb = pd.read_csv(COUNTERBALANCE_CSV)
    cb = cb.where(cb.notna(), None)
    mask = cb['subject_id'] == config.subject_id
    df = cb.loc[mask]

    localizer_trials = parse_blocks(df, df["trial_type"].isin(["word", "image"]))
    training_trials = parse_blocks(df, df["trial_type"].str.contains("training"))
    main_trials = parse_blocks(df, df["trial_type"].eq("test"))

    var_names = list(df.columns) + ['timestamp', 'response', 'rt', 'correct']
    
    """ INITIALIZE EXPERIMENT """
    exp = design.Experiment(name='Pushmi', background_colour=C_BLACK, foreground_colour=C_WHITE)
    exp.add_data_variable_names(var_names)
    control.initialize(exp)

    """ STIMULI """
    w, h = exp.screen.size
    pulse_position = (w//2 - 50, -h//2 + 50)
    pulse = preload(stimuli.Rectangle(size=(50, 50), position=pulse_position))
    images = {name: picture(name, SIZES[name]) for name in STIMS + SHAPES_TRAINING}
    words = {name: word(name) for name in SENTENCE_STIMS}
    fixation = preload(stimuli.Circle(radius=2.5 * SCALE_FACTOR, position=ORIGIN, colour=C_GREY))
    feedback = {label: preload(stimuli.Rectangle(size=(200, 100), position=ORIGIN, colour=colour))
                for label, colour in (('timeout', LIGHTGRAY), ('correct', GREEN), ('incorrect', RED))}
    instructions = [preload(picture(f"instr_{i}", 0.8)) for i in range(1, 7)]
    pause_message = preload(
        stimuli.TextScreen("Pause", "Prenez un moment pour vous reposer",
                           position = (OFFSET_X, -300), heading_size = 50, text_size = 30))
    end_message = preload(
        stimuli.TextScreen("Bravo, vous avez terminé !", "Merci d'avoir participé à cette expérience !",
                           position = (OFFSET_X, -300), heading_size = 50, text_size = 30))

    """ RUN EXPERIMENT """
    try:
        control.start(subject_id=config.subject_id)

        if hardware.eyetracker:
            hardware.calibrate_eyetracker(exp.keyboard)

        if config.localizer:
            show(instructions[0])
            for block in localizer_trials:
                for params in block:
                    run_localizer_trial(**params)
            take_break(pause_message)

        if config.training:
            show(instructions[1])
            show(instructions[2])
            for block_number, block in enumerate(training_trials):
                for params in block:
                    run_main_trial(**params)
                if block_number == 0:
                    show(instructions[3])
                elif block_number == 1:
                    show(instructions[4])
            take_break(pause_message)

        show(instructions[5])
        for block_number, block in enumerate(main_trials, 1):
            for params in block:
                run_main_trial(**params)
            if block_number != N_BLOCKS:
                take_break(pause_message)
        show(end_message)
        
    finally:
        hardware.cleanup()
        control.end()