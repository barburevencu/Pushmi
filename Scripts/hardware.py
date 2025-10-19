from expyriment import io
from expyriment.misc.constants import C_GREY, C_WHITE, K_f, K_j, K_SPACE
import time
from constants import *

class ResponseMEG:
    """
    An Expyriment class to handle responses from MEG button boxes via parallel ports.
    """

    def __init__(self):
        """
        Initializes the parallel ports for response collection.
        """
        self.buttons = BUTTONS
        self.port_addresses = {'port1': '/dev/parport0', 'port2': '/dev/parport1', 'port3': '/dev/parport2'} 
        self.ports, self.baselines, self.last_values = self._init_ports()

    def _init_ports(self):
        ports = {}
        baselines = {}
        last_values = {}
        for name, address in self.port_addresses.items():
            port = io.ParallelPort(address=address)
            ports[name] = port
            baseline = port.read_status()
            baselines[name] = baseline
            last_values[name] = baseline
        return ports, baselines, last_values

    def check_response(self):
        """
        Checks if a button has been pressed on any single configured port.
        """
        for name, port in self.ports.items():
            current_value = port.read_status()
            if current_value != self.last_values[name]:
                self.last_values[name] = current_value
                if current_value != self.baselines[name]:
                    return self.buttons.get(name, {}).get(current_value)
        return None

    def wait(self, buttons=['LR', 'RR'], duration=float('inf')):
        t0 = time.time() * 1000  # Convert to milliseconds

        for port in self.ports.values():
            port.read_status()

        while time.time() * 1000 - t0 <= duration:
            button = self.check_response()
            if button in buttons:
                rt = time.time() * 1000 - t0
                return button, rt

        return None, None

class HardwareManager:
    """Manages MEG, eyetracker, and keyboard hardware."""
    
    def __init__(self, subject_id, meg=False, eyetracker=False):
        """
        Initialize hardware manager.
        """
        self.subject_id = subject_id
        self.use_meg = meg
        self.use_eyetracker = eyetracker
        self.meg_handler = None
        self.eyetracker = None
        self.response_keys = None
        self.port = None
        
    def setup(self):
        """Initialize hardware based on configuration."""
        if self.use_meg:
            self.meg_handler = ResponseMEG()
            self.port = self.meg_handler.ports['port2']
            self.response_keys = ['LR', 'RR']
        else:
            self.response_keys = [K_f, K_j]
        
        if self.use_eyetracker:
            import eyelink
            cfg = {
                'edf_file_base_name': f"s{self.subject_id:02}",
                'background_color': C_GREY,
                'foreground_color': C_WHITE,
                'calibration_target_size': 100
            }
            self.eyetracker = eyelink.EyeLinkWrapper(cfg)
            self.eyetracker.initialize()
        
        return self
    
    def calibrate_eyetracker(self, exp_keyboard):
        """Calibrate eyetracker if available."""
        if self.eyetracker:
            self.eyetracker.calibrate()
            exp_keyboard.wait(K_SPACE)
            self.eyetracker.start_new_block()
            self.eyetracker.start_recording()
    
    def cleanup(self):
        """Cleanup hardware resources."""
        if self.eyetracker:
            self.eyetracker.stop_recording()
            self.eyetracker.close()