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

def picture(filename, scale_factor):
    """Load and scale an image stimulus."""
    stim = stimuli.Picture(str(STIM_DIR / f"{filename}.png"), position=ORIGIN)
    stim.scale(scale_factor)
    return preload(stim)

def word(name):
    """Load a word stimulus, applying accent corrections if needed."""
    return preload(stimuli.TextLine(ACCENTS.get(name, name), position=ORIGIN, text_size=TEXTSIZE))

def log(event):
    logger.info(f"Event: {event}; key: {TRIGGERS[event]}.")
    if hardware.meg_handler:
        hardware.port.set_data(TRIGGERS.get(event, 0))
    if hardware.eyetracker:
        hardware.eyetracker.send_message(event)

def draw_flip(*stims, event=None):
    exp.screen.clear()
    for s in stims:
        s.present(False, False)
    fixation_dot.present(False, False)
    exp.screen.update()
    if event: log(event)

def draw_pulse(*stims, event=None):
    t0 = exp.clock.time
    draw_flip(*stims, pulse, event=event)
    exp.clock.wait(50 - (exp.clock.time - t0))

def draw_wait_time(*stims, duration, pulse=True, event=None):
    t0 = exp.clock.time
    if pulse:
        draw_pulse(*stims, event=event)
        draw_flip(*stims, event="baseline")
    else:
        draw_flip(*stims)
    remaining = max(0, duration - (exp.clock.time - t0))
    exp.clock.wait(max(0, remaining))
    #logger.info(f"{event} duration: {exp.clock.time - t0} ms")

def draw_wait_keys(*stims, keys, pulse=True, event=None, timeout=float('inf')):
    t0 = exp.clock.time
    if pulse:
        draw_pulse(*stims, event=event)
        draw_flip(*stims, event="baseline")
    else:
        draw_flip(*stims)
    remaining = max(0, timeout - (exp.clock.time - t0))
    key, rt = exp.keyboard.wait(keys, remaining)
    logger.info(f"{event} duration: {exp.clock.time - t0} ms")
    return KEYMAP.get(key), rt

def draw_wait_buttons(*stims, buttons, pulse=True, event=None, timeout=float('inf')):
    t0 = exp.clock.time
    if pulse:
        draw_pulse(*stims, event=event)
        draw_flip(*stims, event="baseline")
    else:
        draw_flip(*stims)
    remaining = max(0, timeout - (exp.clock.time - t0))
    button, rt = hardware.meg_handler.wait(buttons, remaining)
    logger.info(f"{event} duration: {exp.clock.time - t0} ms")
    return KEYMAP.get(button), rt

def draw_fixation(duration, event='fixation'):
    """Display fixation cross for a given duration."""
    draw_wait_time(duration=duration, event=event)

def draw_assignment(shape1, label1, shape2, label2, order, t = 1000):
    """Draw the assignment phase."""
    shape_first = order == "shape_first"
    stim1, stim2 = (shape1, label1) if shape_first else (label1, shape1)
    stim3, stim4 = (shape2, label2) if shape_first else (label2, shape2)
    event1 = ('assign_shape_1', 'assign_label_1') if shape_first else ('assign_label_1', 'assign_shape_1')
    event2 = ('assign_shape_2', 'assign_label_2') if shape_first else ('assign_label_2', 'assign_shape_2')

    draw_fixation(duration=t, event='fix_init')

    draw_wait_time(stim1, duration=t, event=event1[0])
    draw_wait_time(stim2, duration=t, event=event1[1])

    draw_fixation(duration=t, event='fix_inter_assign')

    draw_wait_time(stim3, duration=t, event=event2[0])
    draw_wait_time(stim4, duration=t, event=event2[1])

    draw_fixation(duration=t, event='fix_inter_assign')

def draw_locations(central_shape, lateral_shape, side, animate, central_x_final):
    """Draw the assignment phase."""
    offset = -SHAPE_WIDTH if side == 'left' else SHAPE_WIDTH
    lateral_shape.move((offset, 0))

    draw_wait_time(central_shape, duration=1000, event='central_flash')

    if animate:
        draw_wait_time(central_shape, lateral_shape, duration=200, event='lateral_flash')
        move_pair(central_shape, lateral_shape, central_x_final)

    else:
        draw_fixation(duration=800, event='fix_inter_flash')
        draw_wait_time(lateral_shape, duration=200, event='lateral_flash')
        draw_fixation(duration=800, event='fix_post_flash')

def draw_outcome(central_shape, lateral_shape, central_x_final, lateral_x_final):
    """Draw the outcome phase."""
    central_shape.reposition((central_x_final, 0))
    lateral_shape.reposition((lateral_x_final, 0))

    draw_wait_time(central_shape, lateral_shape, duration=200, event='outcome')
    draw_fixation(duration=500, event='fix_post_outcome')

def show_instructions(instruction_stim):
    """Display instruction screen and wait for spacebar."""
    draw_wait_keys(instruction_stim, pulse=False, keys=K_SPACE)

def run_localizer_trial(**params):
    """Run a single localizer trial."""
    t0 = exp.clock.time
    stim = images[params['shape1']] if params['shape1'] else words[params['label1']]

    if hardware.meg_handler:
        key, rt = draw_wait_buttons(stim, timeout=800, buttons=response_buttons, event=params['trial_type'])
    else:
        key, rt = draw_wait_keys(stim, timeout=800, keys=response_keys, event=params['trial_type'])

    if key != params['correct_key']:
        draw_wait_time(feedback['incorrect'], stim, pulse=False, duration=100)
        draw_flip(stim)
    
    params.update(timestamp=t0, response=key, rt=rt, correct=(key == params['correct_key']))
    exp.data.add([params.get(k) for k in var_names])
    exp.clock.wait(max(0, 1000 - (exp.clock.time - t0)))

def final_positions(movement, lateral_position):
    """Return (central_x, lateral_x) given movement direction and lateral side."""
    right = (DISPLACEMENT + HALFWIDTH + OFFSET_X, DISPLACEMENT - HALFWIDTH + OFFSET_X)
    left = (-(DISPLACEMENT - HALFWIDTH) + OFFSET_X, -(DISPLACEMENT + HALFWIDTH) + OFFSET_X)
    pos = right if movement == "right" else left
    return pos if lateral_position == "left" else pos[::-1]

def move_pair(central, lateral, target_x, speed=3):
    """Slide both shapes horizontally until target_x reached."""
    current_x = central.position[0]
    step = speed if target_x > current_x else -speed
    while (target_x - current_x) * step > 0:
        central.move((step, 0))
        lateral.move((step, 0))
        current_x = central.position[0]
        if target_x != current_x:
            draw_flip(central, lateral)

def run_main_trial(**params):
    """Run a single main experiment trial."""
    t0 = exp.clock.time
    
    shape1, shape2 = images[params['shape1']], images[params['shape2']]
    label1, label2 = words[params['label1']], words[params['label2']]
    central_shape = shape1 if params['central_shape'] == 'shape1' else shape2
    lateral_shape = shape2 if params['central_shape'] == 'shape1' else shape1
    
    central_x_final, lateral_x_final = final_positions(params['movement'], params['lateral_position'])

    draw_assignment(shape1, label1, shape2, label2, order=params['assignment_order'])
    draw_locations(central_shape, lateral_shape, 
                  side=params['lateral_position'],
                  animate='assignment' in params['trial_type'],
                  central_x_final=central_x_final)
    draw_outcome(central_shape, lateral_shape, central_x_final, lateral_x_final)
    
    # Test sentence
    for i, word in enumerate(params['test_sentence'].split(), 1):
        draw_wait_time(words[word], duration=400, event=f'test_{i}')
    
    if hardware.meg_handler:
        key, rt = draw_wait_buttons(timeout=1800, buttons=response_buttons, event='response')
    else:
        key, rt = draw_wait_keys(timeout=1800, keys=response_keys, event='response')

    fb_key = "timeout" if rt is None else ("correct" if key == params['correct_key'] else "incorrect")
    
    draw_wait_time(feedback[fb_key], duration=200, event=f"fb_{fb_key}")
    
    # Reset shape parameters
    shape1.reposition(ORIGIN)
    shape2.reposition(ORIGIN)

    # Add data to experiment data
    params.update(timestamp=t0, response=key, rt=rt, correct=(key == params['correct_key']))
    exp.data.add([params.get(k) for k in var_names])

def take_break(message):
    """Display break message and wait for spacebar."""
    if hardware.eyetracker:
        hardware.eyetracker.stop_recording()
    show_instructions(message)
    if hardware.eyetracker:
        hardware.eyetracker.start_new_block()
        hardware.eyetracker.start_recording()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=int, default=2)
    args, extras = parser.parse_known_args()
    
    subject_id, flags = args.subject_id, set(extras)
    
    if 'fullscreen' in flags:
        OFFSET_X, ORIGIN = -330, (-330, 0)
        SCALE_INSTR = 0.8
        logger.disabled = True
        io.defaults.datafile_directory = io.defaults.eventfile_directory = f"../Data/Pilot/Behavior/sub-{subject_id:02d}"
    else:
        OFFSET_X, ORIGIN = 0, (0, 0)
        SCALE_INSTR = 0.5
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

    pulse = preload(stimuli.Rectangle((50, 50), position=(w//2 - 50, -h//2 + 50)))
    images = {name: picture(name, SIZES[name]) for name in STIMS + SHAPES_TRAINING}
    words = {name: word(name) for name in SENTENCE_STIMS}
    fixation_dot = preload(stimuli.Circle(2.5 * SCALE_FACTOR, position=ORIGIN, colour=C_GREY))
    
    instructions = {name: preload(picture(f"instr_{num}", SCALE_INSTR)) for name, num in INSTRUCTIONS.items()}
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
        show_instructions(instructions["localizer"])
        for block in localizer_trials:
            for params in block:
                run_localizer_trial(**params)
        take_break(pause_message)
    
    if 'training' in flags:
        show_instructions(instructions["training_intro"])
        show_instructions(instructions["training_assignment"])
        for block_number, block in enumerate(training_trials):
            for params in block:
                run_main_trial(**params)
            if block_number == 0:
                show_instructions(instructions["training_no_animation"])
            elif block_number == 1:
                show_instructions(instructions["training_animation"])
        take_break(pause_message)
    
    show_instructions(instructions["main_experiment"])
    for block_number, block in enumerate(main_trials, 1):
        for params in block:
            run_main_trial(**params)
        if block_number != N_BLOCKS:
            take_break(pause_message)
    show_instructions(end_message)
    
    hardware.cleanup()
    control.end()