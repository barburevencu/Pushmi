from expyriment import control, io
from expyriment.misc.constants import K_ESCAPE

# This dictionary maps the raw port codes to human-readable button names.
BUTTONS = {
    'port1_4': 'LR',
    'port1_8': 'LG',
    'port1_2': 'LY',
    'port2_2': 'RR',
    'port2_4': 'RG',
    'port2_8': 'RY',
    'port1_1': 'RB',
    'port3_8': 'LB',
}

# This dictionary maps event names to integer trigger codes for MEG.
TRIGGERS = {
    'initial_fixation': 1,
    'assignment_1_shape': 2,
    'assignment_1_label': 3,
    'inter_assignment_fixation': 4,
    'assignment_2_shape': 5,
    'assignment_2_label': 6,
    'post_assignment_fixation': 7,
    'central_location_flash': 8,
    'post_central_fixation': 9,
    'peripheral_location_flash': 10,
    'post_peripheral_fixation': 11,
    'outcome': 12,
    'post_outcome_fixation': 13,
    'test_sentence_1': 14,
    'test_sentence_2': 15,
    'test_sentence_3': 16,
    'test_sentence_4': 17,
    'test_sentence_5': 18,
    'response': 19,
    'feedback_correct': 20,
    'feedback_incorrect': 21,
    'feedback_timeout': 22,
    'word': 23,
    'image': 24
}

class responseMEG(object):
    """
    An Expyriment class to handle responses from MEG button boxes via parallel ports.
    """
    def __init__(self, exp, port_addresses):
        """
        Initializes the parallel ports for response collection.
        """
        self.ports = {}
        self.baselines = {}
        self.last_values = {}
        self.exp = exp
        self.buttons = BUTTONS     

        for name, address in port_addresses.items():
            self.ports[name] = io.ParallelPort(address=address)
            
            baseline = self.ports[name].read_status()
            self.baselines[name] = baseline
            self.last_values[name] = baseline

    def check_response(self):
        """
        Checks if a button has been pressed on any single configured port.
        """
        current_reads = {name: port.read_status() for name, port in self.ports.items()}
        changes = {name: current_reads[name] - self.baselines[name] for name in self.ports}
        active_ports = [name for name, change in changes.items() if change != 0]
        
        button = None
        
        if len(active_ports) == 1:
            port_name = active_ports[0]
            current_value = current_reads[port_name]
            
            if current_value != self.last_values[port_name]:
                button = f'{port_name}_{current_value}'

        for name, value in current_reads.items():
            if value != self.last_values[name]:
                self.last_values[name] = value
                
        return button

    def wait(self, correct_key, duration=float('inf')): 
        """
        Waits for a response and determines if it was correct.
        """
        t0 = self.exp.clock.time
        
        for port in self.ports.values():
            port.read_status()

        while True:
            raw_response = self.check_response()
            
            if raw_response:
                rt = self.exp.clock.time - t0
                
                button_pressed = self.buttons.get(raw_response)

                correct = (button_pressed == correct_key)

                return button_pressed, rt, correct

            if self.exp.clock.time - t0 > duration:
                correct = False if correct_key else True
                return None, None, correct 

            if self.exp.keyboard.check(K_ESCAPE):
                control.end()