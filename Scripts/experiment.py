import pandas as pd, argparse
from expyriment import control, design, stimuli, io
from expyriment.misc.constants import C_BLACK, C_WHITE, C_GREY, K_f, K_j, K_SPACE
from constants import *
from hardware import HardwareManager
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger()

def preload(stimulus):
    stimulus.preload()
    return stimulus

def picture(name, scale_factor):
    """Load and scale an image stimulus."""
    stim = stimuli.Picture(str(STIM_DIR / f"{name}.png"), position=ORIGIN)
    stim.scale(scale_factor)
    return preload(stim)

def word(name):
    """Load a word stimulus, applying accent corrections if needed."""
    return preload(stimuli.TextLine(ACCENTS.get(name, name), position=ORIGIN, text_size=TEXTSIZE))

def present(*stims, duration=None, keys=None, buttons=None, event=None):
    """Present stimuli and optionally wait for response."""
    t0 = exp.clock.time
    exp.screen.clear()
    for s in stims:
        s.present(False, False)
    fixation.present(False, True)

    if hardware.meg_handler and hardware.port and event:
        hardware.port.set_data(TRIGGERS[event])
    if hardware.eyetracker and event:
        hardware.eyetracker.send_message(event)

    if not (duration or keys or buttons):
        return None, None

    remaining = None if duration is None else max(0, duration - (exp.clock.time - t0))

    if buttons and hardware.meg_handler:
        key, rt = hardware.meg_handler.wait(buttons, remaining)
        return KEYMAP.get(key), rt
    if keys:
        key_list = keys if isinstance(keys, list) else [keys]
        key, rt = exp.keyboard.wait(key_list, remaining)
        return KEYMAP.get(key), rt
    if remaining:
        exp.clock.wait(remaining)
    return None, None

def draw(*stims, duration=None, keys=None, buttons=None, event=None):
    """Draw stimuli with pulse and collect responses."""
    t0 = exp.clock.time
    present(*stims, pulse, duration=50, event=event)
    key, rt = present(*stims, duration=duration-50, keys=keys, buttons=buttons, event='baseline')
    logger.info(f"{event} duration: {exp.clock.time - t0} ms")
    return key, rt

def draw_seq(steps):
    """Execute a sequence of drawing steps."""
    for stims, dur, evt in steps:
        if not stims:
            draw(duration=dur, event=evt)
        elif isinstance(stims, (list, tuple)):
            draw(*stims, duration=dur, event=evt)
        else:
            draw(stims, duration=dur, event=evt)

def run_localizer_trial(**params):
    """Run a single localizer trial."""
    t0 = exp.clock.time
    stim = images[params['shape1']] if params['shape1'] else words[params['label1']]
    key, rt = draw(stim, duration=800, keys=response_keys, buttons=response_buttons, event=params['trial_type'])
    
    if key != params['correct_key']:
        present(feedback['incorrect'], stim, duration=100)
        present(stim)
    
    params.update(timestamp=t0, response=key, rt=rt, correct=(key == params['correct_key']))
    exp.data.add([params.get(k) for k in var_names])
    exp.clock.wait(max(0, 1000 - (exp.clock.time - t0)))

def final_positions(movement, lateral_position):
    """Return (central_x, lateral_x) given movement direction and lateral side."""
    right = (DISPLACEMENT + HALFWIDTH + OFFSET_X, DISPLACEMENT - HALFWIDTH + OFFSET_X)
    left = (-(DISPLACEMENT - HALFWIDTH) + OFFSET_X, -(DISPLACEMENT + HALFWIDTH) + OFFSET_X)
    pos = right if movement == "right" else left
    return pos if lateral_position == "left" else pos[::-1]

def assignment_seq(order, shape1, label1, shape2, label2):
    """Build the assignment sequence based on whether shape or label comes first."""
    if order == "shape_first":
        return [(None, INITIAL_T, "fix_init"), 
                (shape1, ASSIGNMENT_T, "assign_1_shape"), (label1, ASSIGNMENT_T, "assign_1_label"),
                (None, POST_ASSIGN_T, "fix_inter_assign"), 
                (shape2, ASSIGNMENT_T, "assign_2_shape"), (label2, ASSIGNMENT_T, "assign_2_label"),
                (None, POST_ASSIGN_T, "fix_post_assign")]
    return [(None, INITIAL_T, "fix_init"), 
            (label1, ASSIGNMENT_T, "assign_1_label"), (shape1, ASSIGNMENT_T, "assign_1_shape"),
            (None, POST_ASSIGN_T, "fix_inter_assign"), 
            (label2, ASSIGNMENT_T, "assign_2_label"), (shape2, ASSIGNMENT_T, "assign_2_shape"),
            (None, POST_ASSIGN_T, "fix_post_assign")]

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
    """Run a single main experiment trial."""
    t0 = exp.clock.time
    
    shape1, shape2 = images[params['shape1']], images[params['shape2']]
    label1, label2 = words[params['label1']], words[params['label2']]
    central_shape = shape1 if params['central_shape'] == 'shape1' else shape2
    lateral_shape = shape2 if params['central_shape'] == 'shape1' else shape1
    
    offset = -SHAPE_WIDTH if params['lateral_position'] == 'left' else SHAPE_WIDTH
    central_x, lateral_x = final_positions(params['movement'], params['lateral_position'])
    
    draw_seq(assignment_seq(params['assignment_order'], shape1, label1, shape2, label2))
    lateral_shape.move((offset, 0))

    if params['trial_type'] in ['test', 'training_no_animation']:
        draw_seq([(central_shape, CENTRAL_FLASH_T, 'central_flash'), (None, POST_FLASH_T, 'fix_inter_flash'),
                  (lateral_shape, LATERAL_FLASH_T, 'lateral_flash'), (None, POST_FLASH_T, 'fix_post_flash')])
    else:
        draw_seq([(central_shape, CENTRAL_FLASH_T, 'central_flash'), 
                  ([central_shape, lateral_shape], CENTRAL_FLASH_T, 'central_flash')])
        move_pair_to_x(central_shape, lateral_shape, central_x)

    central_shape.reposition((central_x, 0))
    lateral_shape.reposition((lateral_x, 0))
    draw_seq([([central_shape, lateral_shape], OUTCOME_T, 'outcome'), (None, POST_OUTCOME_T, 'fix_post_outcome')])
    
    for i, word in enumerate(params['test_sentence'].split(), 1):
        draw(words[word], duration=WORD_T, event=f'test_{i}')
    
    key, rt = draw(duration=RESPONSE_T, keys=response_keys, buttons=response_buttons, event='response')
    shape1.reposition(ORIGIN)
    shape2.reposition(ORIGIN)
    
    fb_key = "timeout" if rt is None else ("correct" if key == params['correct_key'] else "incorrect")
    params.update(timestamp=t0, response=key, rt=rt, correct=(key == params['correct_key']))
    exp.data.add([params.get(k) for k in var_names])
    draw(feedback[fb_key], duration=FEEDBACK_T, event=f"fb_{fb_key}")

def take_break(message):
    """Display break message and wait for spacebar."""
    if hardware.eyetracker:
        hardware.eyetracker.stop_recording()
    present(message, keys=K_SPACE)
    if hardware.eyetracker:
        hardware.eyetracker.start_new_block()
        hardware.eyetracker.start_recording()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=int, default=2)
    args, extras = parser.parse_known_args()
    
    subject_id, flags = args.subject_id, set(extras)
    
    if 'fullscreen' in flags:
        logger.disabled = False
        io.defaults.datafile_directory = io.defaults.eventfile_directory = f"../Data/Pilot/Behavior/sub-{subject_id:02d}"
    else:
        control.set_develop_mode(skip_wait_methods=True)

    hardware = HardwareManager(subject_id, 'meg' in flags, 'eyetracker' in flags).setup()
    response_keys, response_buttons = [K_f, K_j], hardware.response_keys if hardware.meg_handler else None

    df = pd.read_csv(COUNTERBALANCE_CSV).where(lambda x: x.notna(), None)
    df = df[df['subject_id'] == subject_id]
    
    localizer_trials = [g.to_dict("records") for _, g in df[df["trial_type"].isin(["word", "image"])].groupby("block_number", sort=True)]
    training_trials = [g.to_dict("records") for _, g in df[df["trial_type"].str.contains("training")].groupby("block_number", sort=True)]
    main_trials = [g.to_dict("records") for _, g in df[df["trial_type"] == "test"].groupby("block_number", sort=True)]
    var_names = list(df.columns) + ['timestamp', 'response', 'rt', 'correct']
    
    exp = design.Experiment('Pushmi', background_colour=C_BLACK, foreground_colour=C_WHITE)
    exp.add_data_variable_names(var_names)
    control.initialize(exp)

    w, h = exp.screen.size
    instructions = {name: preload(picture(f"instr_{num}", 0.8)) for name, num in INSTRUCTIONS.items()}
    pulse = preload(stimuli.Rectangle((50, 50), position=(w//2 - 50, -h//2 + 50)))
    images = {name: picture(name, SIZES[name]) for name in STIMS + SHAPES_TRAINING}
    words = {name: word(name) for name in SENTENCE_STIMS}
    fixation = preload(stimuli.Circle(2.5 * SCALE_FACTOR, position=ORIGIN, colour=C_GREY))
    feedback = {label: preload(stimuli.Rectangle((200, 100), position=ORIGIN, colour=colour))
                for label, colour in [('timeout', LIGHTGRAY), ('correct', GREEN), ('incorrect', RED)]}
    pause_message = preload(stimuli.TextScreen("Pause", "Prenez un moment pour vous reposer",
                           position=(OFFSET_X, -300), heading_size=50, text_size=30))
    end_message = preload(stimuli.TextScreen("Bravo, vous avez terminé !", 
                          "Merci d'avoir participé à cette expérience !",
                          position=(OFFSET_X, -300), heading_size=50, text_size=30))

    control.start(subject_id=subject_id)
    
    if hardware.eyetracker:
        hardware.calibrate_eyetracker(exp.keyboard)
    
    if 'localizer' in flags:
        present(instructions["localizer"], keys=K_SPACE)
        for block in localizer_trials:
            for params in block:
                run_localizer_trial(**params)
        take_break(pause_message)
    
    if 'training' in flags:
        present(instructions["training_intro"], keys=K_SPACE)
        present(instructions["training_assignment"], keys=K_SPACE)
        for block_number, block in enumerate(training_trials):
            for params in block:
                run_main_trial(**params)
            if block_number == 0:
                present(instructions["training_no_animation"], keys=K_SPACE)
            elif block_number == 1:
                present(instructions["training_animation"], keys=K_SPACE)
        take_break(pause_message)
    
    present(instructions["main_experiment"], keys=K_SPACE)
    for block_number, block in enumerate(main_trials, 1):
        for params in block:
            run_main_trial(**params)
        if block_number != N_BLOCKS:
            take_break(pause_message)
    present(end_message, keys=K_SPACE)
    
    hardware.cleanup()
    control.end()