import os
import pandas as pd
import numpy as np
import argparse

# Conditionally import hardware-specific modules
try:
    import eyelink
    from meg import responseMEG, trigger_labels
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("Warning: MEG or EyeLink libraries not found. Running in simulation mode.")

from expyriment import design, control, stimuli, io, misc
from constants import *
from counterbalancing import counterbalance

class Experiment:
    def __init__(self, subject_id = 1, localizer=False, fullscreen=False, training=False, meg=False, eyetracking=False):
        self.subject_id = subject_id
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'Randomization', "counterbalancing.csv")

        self.df = pd.read_csv(csv_path)
        self.df = self.df.loc[self.df['ID'] == subject_id]

        self.localizer = localizer
        self.training = training
        self.meg = meg and HARDWARE_AVAILABLE
        self.eyetracking = eyetracking and HARDWARE_AVAILABLE

        # 1. Initialize Expyriment
        self._initialize_expyriment(fullscreen)

        # 2. Initialize hardware (MEG/EyeLink) if enabled
        self._initialize_hardware()

        # 3. Load and prepare trial data
        self._load_and_prepare_trials()

        # 4. Preload all stimuli for efficiency
        self._preload_stimuli()

        # 5. Preload instruction screen
        self._preload_instructions()

        self.exp.keyboard.set_quit_key(misc.constants.K_q)

    def _initialize_expyriment(self, fullscreen):
        """Sets up the Expyriment environment, window, and data logging."""
        control.defaults.window_mode = not fullscreen
        #control.defaults.window_size = (2560, 1600)
        control.defaults.background_colour = misc.constants.C_BLACK
        control.defaults.foreground_colour = misc.constants.C_WHITE
        #control.defaults.opengl = OPENGL

        self.exp = design.Experiment(name="Pushmi")
        #control.set_develop_mode(skip_wait_methods=True)
        control.initialize(self.exp)

        

        # Define all data headers and add to Expyriment data object
        self.data_headers = list(self.df.columns) + [
            'timestamp', 'participant_response', 'reaction_time', 'participant_correct'
        ]
        self.exp.add_data_variable_names(self.data_headers)

    def _initialize_hardware(self):
        """Initializes EyeLink and MEG connections if they are enabled."""
        if self.eyetracking:
            cfg_eyelink = {
                'edf_file_base_name': f"s{self.subject_id:02}",
                'background_color': GRAY,
                'foreground_color': WHITE,
                'calibration_target_size': 100
            }
            self.et = eyelink.EyeLinkWrapper(cfg_eyelink)
        else:
            self.et = None

        if self.meg:
            port_addresses = {'port1': '/dev/parport0', 'port2': '/dev/parport1', 'port3': '/dev/parport2'} 
            response_mappings_meg = {'LR': 'f', 'RR': 'j'}
            self.meg_handler = responseMEG(self.exp, port_addresses, response_mappings_meg)
            self.port = self.meg_handler.ports['port2'] # Primary port for triggers
        else:
            self.meg_handler = None
            self.port = None

    def _load_and_prepare_trials(self):
        """Loads the counterbalanced trial data and splits it into different phases."""
        self.trials_localizer = self.df[self.df['trial_type'] == 'localizer'].to_dict('records')
        self.trials_training = self.df[self.df['trial_type'].str.contains('training', na=False)].to_dict('records')
        self.trials_exp = self.df[self.df['trial_type'] == 'test'].to_dict('records')

    def _preload_stimuli(self):
        """Creates and preloads all visual and text stimuli to optimize trial performance."""
        # --- Static Screens ---
        self.pause_screen = stimuli.TextScreen(
            "Pause", "Prenez un moment pour vous reposer", 
            position = (-330, -300), 
            heading_size = 50, heading_colour = WHITE, 
            text_size = 30, text_colour = WHITE)

        # --- Fixation Cross on its own dedicated canvas for efficiency ---
        self.fixation = stimuli.Circle(radius = 2.5 * SCALE_FACTOR, position = ORIGIN, line_width=0, colour=misc.constants.C_GREY)

        self.fixation_canvas = stimuli.Canvas(size=self.exp.screen.size)
        self.fixation.plot(self.fixation_canvas)
        self.fixation_canvas.preload()

        feedback_size = (200, 100)
        feedback_pos = ORIGIN

        self.timeout_rectangle = stimuli.Rectangle(size=feedback_size, position = feedback_pos, colour=LIGHTGRAY)
        self.incorrect_rectangle = stimuli.Rectangle(size=feedback_size, position = feedback_pos, colour=RED)
        self.correct_rectangle = stimuli.Rectangle(size=feedback_size, position = feedback_pos, colour=GREEN)

        # --- Diode Square for timing validation ---
        self.white_square = stimuli.Rectangle(size=(50, 50), colour=misc.constants.C_WHITE, position=(self.exp.screen.size[0]/2 - 50, -self.exp.screen.size[1]/2 + 50))
        self.black_square = stimuli.Rectangle(size=(1, 1), colour=misc.constants.C_BLACK, position=(self.exp.screen.size[0]/2 - 50, -self.exp.screen.size[1]/2 + 50))

        # --- Main Canvas for dynamic stimuli ---
        self.main_canvas = stimuli.Canvas(size=self.exp.screen.size)
        
        # --- Image Stimuli (Shapes and Objects) ---
        image_names = ['cercle', 'losange', 'carré', 'croix', 'étoile', 'pentagone', 'biche', 'dinde', 'lampe', 'pince']
        self.image_stimuli = {name: self._create_image(f"{name}.png") for name in image_names}

        # --- Text Stimuli ---
        unique_words = set(self.df['label1'].dropna()) | set(self.df['label2'].dropna())
        sentences = self.df['test_sentence'].dropna().str.split(' ').explode()
        unique_words.update(sentences)
        self.text_stimuli = {word: stimuli.TextLine(text=word, position = ORIGIN, text_colour=WHITE, text_size=TEXTSIZE) for word in unique_words if word}

        # --- Preload all created stimuli ---
        for stim in [self.fixation, self.pause_screen, 
            self.timeout_rectangle, self.incorrect_rectangle, self.correct_rectangle,
            self.main_canvas, self.white_square, self.black_square] + \
            list(self.image_stimuli.values()) + list(self.text_stimuli.values()):
            stim.preload()

    def _preload_instructions(self):
        dir_path = os.path.join(os.path.dirname(__file__), '..', 'Stimuli', 'Instructions')
        self.instruction_screens = {}

        all_files = os.listdir(dir_path)
        image_files = sorted([f for f in all_files if f.startswith("Instructions") and f.endswith(".png")])

        for filename in image_files:
            image_path = os.path.join(dir_path, filename)
            screen_name = os.path.splitext(filename)[0] # Get filename without extension

            # 1. Create a Picture stimulus from the image file
            instruction_image = stimuli.Picture(image_path, position=ORIGIN)
            instruction_image.scale((0.8, 0.8))
            # Preload the raw image data
            instruction_image.preload()

            # 2. Create a Canvas the size of the experiment screen
            instruction_canvas = stimuli.Canvas(size=self.exp.screen.size)

            # 3. Plot the image onto the canvas, scaled to fit the canvas entirely
            #    By plotting with default arguments, it will scale to fill the canvas.
            instruction_image.plot(instruction_canvas)

            # 4. Preload the final canvas for fast display
            instruction_canvas.preload()

            # Store the preloaded canvas in our dictionary
            self.instruction_screens[screen_name] = instruction_canvas

    def _create_image(self, filename):
        path = os.path.join(os.path.dirname(__file__), '..', 'Stimuli', filename)
        target_size = (IMAGE_WIDTH, IMAGE_WIDTH) \
            if any(obj in filename for obj in ('biche', 'dinde', 'lampe', 'pince')) else (SHAPE_WIDTH, SHAPE_WIDTH)
        
        # 1. Create the picture object from the file path
        picture_stim = stimuli.Picture(path, position=ORIGIN)
        
        # 2. Get the original dimensions of the loaded picture
        original_width, original_height = 1000, 1000
        
        # 3. Calculate the scaling factors for width and height
        scale_x = target_size[0] / original_width
        scale_y = target_size[1] / original_height
        
        # 4. Scale the picture using the calculated factors
        picture_stim.scale((scale_x, scale_y))
        
        return picture_stim

    def get_word(self, label):
        """Retrieves a preloaded TextLine stimulus from the dictionary."""
        return self.text_stimuli[label]

    ### --- CORE LOGIC & HELPER METHODS --- ###
    def _send_event_marker(self, event_desc, trial_info):
        """Sends a trigger to MEG and a message to the eye-tracker."""
        self._log_event(event_desc, trial_info)
        if self.meg and event_desc in trigger_labels:
            self.port.set_data(trigger_labels[event_desc])
        if self.eyetracking:
            self.et.send_message(event_desc)

    def _start_eyetracker(self):
            if self.eyetracking:
                self.et.start_new_block()
                self.et.start_recording()
    
    def _stop_eyetracker(self):
        if self.eyetracking: self.et.stop_recording()
            
    def _log_event(self, description, event_info=None):
        """Formats event info and logs it to Expyriment's event file."""
        message = description
        if event_info:
            details = "; ".join([f"{key}={value}" for key, value in event_info.items() if value is not None])
            message = f"{description} [{details}]"
        self.exp.events.log(message)

    def compute_shape_positions(self, central_shape_identity, peripheral_position, shape1, shape2, movement):
        
        if central_shape_identity == 'shape1':
            central_shape, marginal_shape = shape1, shape2
            if peripheral_position == 'left':
                marginal_shape.position = (central_shape.position[0] - SHAPE_WIDTH/2 - HALFWIDTH, 0)
            else: # right
                marginal_shape.position = (central_shape.position[0] + SHAPE_WIDTH/2 + HALFWIDTH, 0)
        else: # shape2 is central
            central_shape, marginal_shape = shape2, shape1
            if peripheral_position == 'left':
                marginal_shape.position = (central_shape.position[0] - SHAPE_WIDTH/2 - HALFWIDTH, 0)
            else: # right
                marginal_shape.position = (central_shape.position[0] + SHAPE_WIDTH/2 + HALFWIDTH, 0)

        # Set anchor position based on movement direction
        anchor_x = -DISPLACEMENT + ORIGIN[0] if movement == 'left' else DISPLACEMENT + ORIGIN[0]

        anchor_is_marginal = (movement == 'left' and peripheral_position == 'left') or \
                             (movement == 'right' and peripheral_position == 'right')

        if anchor_is_marginal:
            marginal_shape_goal = [anchor_x, 0]
            central_shape_goal = [anchor_x + SHAPE_WIDTH if movement == "left" else anchor_x - SHAPE_WIDTH, 0]
        else:
            central_shape_goal = [anchor_x, 0]
            marginal_shape_goal = [anchor_x + SHAPE_WIDTH if movement == "left" else anchor_x - SHAPE_WIDTH, 0]

        return central_shape, marginal_shape, central_shape_goal, marginal_shape_goal

    def _draw_to_canvas(self, stims, include_probe=False):
        t0 = self.exp.clock.time

        (self.white_square if include_probe else self.black_square).present(clear=True, update=False)

        for stim in stims:
            stim.present(clear = False, update = False)
        
        self.fixation.present(clear=False, update=True)
        return self.exp.clock.time - t0

    ### --- PRESENTATION METHODS --- ###
    def _show_stimuli(self, stims, duration, trial_info, event_desc, include_probe=True):
        """Generic stimulus presentation function."""
        
        # PREP
        t0 = self.exp.clock.time
    
        self._draw_to_canvas(stims, include_probe=include_probe)
        self._send_event_marker(event_desc, trial_info)

        lag = self.exp.clock.time - t0
        #print(f"{event_desc} prep: {lag:.2f} ms")

        self.exp.clock.wait(50 - lag)

        if duration > 50:
            self._draw_to_canvas(stims, include_probe=False)
            if self.meg:
                self.port.set_data(0)
            lag = self.exp.clock.time - t0
            self.exp.clock.wait(max(0, duration - lag))

        t = self.exp.clock.time - t0  
        
        print(f"{event_desc} event: {t:.2f} ms")
        return t

    def _show_fixation(self, duration, trial_info, event_desc='fixation'):
        self._show_stimuli([], duration, trial_info, event_desc, include_probe=False)

    def _pause(self):
        self.pause_screen.present()
        self.exp.keyboard.wait(keys=misc.constants.K_SPACE)

    def _show_sentence(self, sentence, trial_info):
        """Presents a sentence word by word."""
        for i, word_text in enumerate(sentence.split()):
            word_stim = self.text_stimuli.get(word_text)
            self._show_stimuli([word_stim], duration=WORD, trial_info=trial_info, event_desc=f'test_sentence_{i+1}')

    def _show_assignment(self, shape, label_text, order, trial_info, assignment_number=1):
        label_stim = self.get_word(label_text)
        
        event_desc1 = f"assignment_{assignment_number}_{'shape' if order == 'symbol_first' else 'label'}"
        event_desc2 = f"assignment_{assignment_number}_{'label' if order == 'symbol_first' else 'shape'}"
        
        first_stim = shape if order == "symbol_first" else label_stim
        second_stim = label_stim if order == "symbol_first" else shape
        
        self._show_stimuli([first_stim], duration=ASSIGNMENT, trial_info=trial_info, event_desc=event_desc1)
        self._show_stimuli([second_stim], duration=ASSIGNMENT, trial_info=trial_info, event_desc=event_desc2)

    def _move_shapes(self, shape1, shape2, central_shape, movement, goal_x, trial_info):
        step = -0.5 * SCALE_FACTOR if movement == 'left' else 0.5 * SCALE_FACTOR
        event_description = f"moving_shapes_{movement}"
        
        # Animate frame by frame
        while abs(central_shape.position[0] - goal_x) > 0:
            shape1.move((step, 0))
            shape2.move((step, 0))
            
            shape1.present(clear=True, update=False)
            shape2.present(clear=False,update=False)
            self.fixation.present(clear=False,update=True)
            
            # Log each frame of movement
            self._log_event(event_description, {**trial_info, 'pos1_x': shape1.position[0], 'pos2_x': shape2.position[0]})

    def _get_response(self, correct_key, trial_info, stims = [], timeout=None, event_desc='response'):
        start_prep = self.exp.clock.time

        self._show_stimuli(stims, 50, trial_info, include_probe=True, event_desc=event_desc)
        self._draw_to_canvas(stims)
        if self.meg:
                self.port.set_data(0)

        end_prep = self.exp.clock.time
        lag = end_prep - start_prep

        if self.meg:
            response_key_char, rt, correct = self.meg_handler.wait(correct_key, duration=timeout-lag)
        
        else:
            key_map = {'f': misc.constants.K_f, 'j': misc.constants.K_j}
            allowed_keys = list(key_map.values())
            
            key, rt = self.exp.keyboard.wait(keys=allowed_keys, duration=timeout-lag)
            
            response_key_char = 'f' if key == misc.constants.K_f else 'j' if key == misc.constants.K_j else None
            correct = response_key_char == correct_key if response_key_char else False
            
            if pd.isna(correct_key) and response_key_char is None:
                correct = True
            
        feedback = (
            self.timeout_rectangle
            if rt is None
            else (self.correct_rectangle if correct else self.incorrect_rectangle)
        )

        return response_key_char, rt, correct, feedback

    def _show_feedback(self, feedback, trial_info, event_desc):
        self._show_stimuli([feedback], 200, trial_info, event_desc, include_probe=True)

    def _show_instructions(self, instructions_canvas):
        instructions_canvas.present()
        self.exp.keyboard.wait(keys=misc.constants.K_SPACE)
        
    ### --- MAIN TRIAL AND EXPERIMENT FLOW --- ###
    def _run_main_trial(self, trial_info):
        shape1 = self.image_stimuli[trial_info['shape1']]
        shape2 = self.image_stimuli[trial_info['shape2']]
        label1 = trial_info['label1']
        label2 = trial_info['label2']
        [s.reposition(ORIGIN) for s in [shape1, shape2]]

        # Decide which shape will be centrally flashed
        shape_first, shape_second = (shape1, shape2) if trial_info['location_order'] == 'shape1_first' else (shape2, shape1)

        trial_info['timestamp'] = self.exp.clock.time

        # INITIAL FIXATION
        self._show_fixation(duration=INITIAL_FIXATION, trial_info=trial_info, event_desc='initial_fixation')

        # ASSIGNMENT 1
        self._show_assignment(shape1, label1, trial_info['assignment_order'], trial_info, 1)

        # INTER-ASSIGNMENT FIXATION
        self._show_fixation(duration=POST_ASSIGNMENT, trial_info=trial_info, event_desc='inter_assignment_fixation')

        # ASSIGNMENT 2
        self._show_assignment(shape2, label2, trial_info['assignment_order'], trial_info, 2)

        # POST-ASSIGNMENT FIXATION
        self._show_fixation(duration=POST_ASSIGNMENT, trial_info=trial_info, event_desc='post_assignment_fixation')

        # Compute positions of shapes for the flashing point
        central_shape, marginal_shape, central_shape_goal, marginal_shape_goal = self.compute_shape_positions(
            trial_info['central_shape'], trial_info['peripheral_position'], shape1, shape2, trial_info['movement'])

        # SHAPE FLASHING
        if trial_info['trial_type'] not in ["test", "training_no_animation"]:
            # For training, show both shapes together at their starting positions
            self._show_stimuli([shape_first], duration=CENTRAL_FLASH, trial_info=trial_info, event_desc='training_show_central_shape')
            self._show_stimuli([shape_first, shape_second], duration=CENTRAL_FLASH, trial_info=trial_info, event_desc='training_show_both_shapes')
            # Then, animate their movement to the final positions
            self._move_shapes(shape1, shape2, central_shape, trial_info['movement'], central_shape_goal[0], trial_info)
        
        else:
            # 1. Flash the first shape (central location) for 1000ms
            self._show_stimuli([shape_first], duration=CENTRAL_FLASH, trial_info=trial_info, event_desc='central_location_flash')

            # 2. Show fixation for 800ms
            self._show_fixation(duration=POST_FLASH, trial_info=trial_info, event_desc='post_central_fixation')

            # 3. Flash the second shape (peripheral location) for 200ms
            self._show_stimuli([shape_second], duration=PERIPHERAL_FLASH, trial_info=trial_info, event_desc='peripheral_location_flash')
            
            # 4. Show fixation again for 800ms
            self._show_fixation(duration=POST_FLASH, trial_info=trial_info, event_desc='post_peripheral_fixation')

            # 5. Internally move shapes to their final goal positions for the subsequent outcome display
            central_shape.reposition(central_shape_goal)
            marginal_shape.reposition(marginal_shape_goal)

        # OUTCOME
        self._show_stimuli([shape1, shape2], duration=OUTCOME, trial_info=trial_info, event_desc='outcome')

        # POST-OUTCOME FIXATION
        self._show_fixation(duration=POST_OUTCOME, trial_info=trial_info, event_desc='post_outcome_fixation')
        
        # TEST SENTENCE
        self._show_sentence(trial_info['test_sentence'], trial_info)    
        
        # RESPONSE
        response_key, rt, correct, feedback = self._get_response(trial_info['correct_key'], trial_info, timeout=2000)

        # FEEDBACK
        self._show_feedback(feedback, trial_info=trial_info, event_desc='feedback')
        
        # --- Record Trial-Level Data ---
        trial_info.update({
            'participant_response': response_key,
            'reaction_time': rt,
            'participant_correct': correct  # You can update this if needed
        })

        self.exp.data.add([trial_info[key] for key in self.data_headers])

    def _run_localizer_trial(self, trial_info,timeout=900):
        start_time = self.exp.clock.time
        trial_info['timestamp'] = start_time
        correct_key = trial_info['correct_key']

        label = trial_info['label1']
        shape = trial_info['shape1']

        if isinstance(label, str):
            stims, event = [self.get_word(label)], 'word'
        elif isinstance(shape, str):
            stims, event = [self.image_stimuli[shape]], 'image'

        response_key, rt, correct, feedback = self._get_response(correct_key, trial_info, stims = stims, timeout=timeout, event_desc=event)
        
        if not correct:
            t = self._draw_to_canvas([self.incorrect_rectangle] + stims)

        lag = self.exp.clock.time - start_time
        remaining = 1000 - lag

        if remaining <= 100:
            self.exp.clock.wait(remaining)
        else:
            self.exp.clock.wait(100)
            t = self._draw_to_canvas(stims)
            self.exp.clock.wait(remaining - 100 - t)

        # Record data
        trial_info.update({
            'participant_response': response_key,
            'reaction_time': rt,
            'participant_correct': correct
        })
        
        self.exp.data.add([trial_info[key] for key in self.data_headers])
        
        #print(f"response duration: {self.exp.clock.time - start_time:.2f} ms")

    def _run_block(self, trials, trial_fn, instructions=[], pause_points=[], mid_instructions={}, restart_interval=None):
        for instr in instructions:
            self._show_instructions(self.instruction_screens[instr])

        self._start_eyetracker()

        for trial_info in trials:
            trial_number = trial_info['trial_number']

            if self.exp.keyboard.check(misc.constants.K_ESCAPE):
                control.end()
                return

            #print("----------")
            #print(trial_number)

            trial_fn(trial_info)

            if trial_number in pause_points:
                self._stop_eyetracker()
                self._pause()
                self._start_eyetracker()

            if trial_number in mid_instructions:
                self._show_instructions(self.instruction_screens[mid_instructions[trial_number]])

            if restart_interval and trial_number % restart_interval == 0:
                self._stop_eyetracker()
                self._pause()
                self._start_eyetracker()

        self._stop_eyetracker()

    def run(self):
        """Start and run the experiment."""
        control.start(subject_id=self.subject_id)

        if self.eyetracking:
            self.et.initialize()
            self.et.calibrate()
            self._pause()
       
        if self.localizer:
            self._run_block(
                self.trials_localizer, 
                trial_fn=self._run_localizer_trial,
                instructions=['Instructions.002'],
                pause_points=[64, 544]
                )
            self._pause()

        if self.training:
            self._run_block(
                self.trials_training,
                trial_fn=self._run_main_trial,
                instructions=['Instructions.003', 'Instructions.004'],
                mid_instructions={6: 'Instructions.005', 16: 'Instructions.006'})
            self._pause()

        self._run_block(
            self.trials_exp, trial_fn=self._run_main_trial, 
            instructions=['Instructions.007'], restart_interval = 32)

        if self.eyetracking: 
            self.et.close()

        control.end(goodbye_text="Experiment finished. Thank you!", goodbye_delay=2000)
    
if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run Pushmi.")
    
    # Positional argument for subject ID, now optional with a default value
    parser.add_argument("subject_id", type=int, nargs='?', default=1, 
                        help="Participant ID (integer). Defaults to 1 if not provided.")
    
    # Optional flags (default to True if not present)
    parser.add_argument("--localizer", action="store_true", help="Include the localizer task.")
    parser.add_argument("--training", action="store_true", help="Include the training block.")
    
    # Parse the arguments from the command line
    args = parser.parse_args()

    # Instantiate the Experiment class with the parsed arguments
    exp = Experiment(
        subject_id=args.subject_id,
        fullscreen=True, 
        localizer=False, #args.localizer
        training=False, #args.training,
        meg=False,
        eyetracking=False
    )
    
    # Run the experiment
    exp.run()