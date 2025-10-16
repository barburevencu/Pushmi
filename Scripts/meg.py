from expyriment import io
import time
from constants import *

class ResponseMEG:
    """
    An Expyriment class to handle responses from MEG button boxes via parallel ports.
    """

    def __init__(self, exp):
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

    def wait(self, buttons=['LR', 'RR'], duration=None):
        t0 = time.time() * 1000  # Convert to milliseconds
        if duration is None: duration = float('inf')

        for port in self.ports.values():
            port.read_status()

        while time.time() * 1000 - t0 <= duration:
            button = self.check_response()
            if button:
                rt = time.time() * 1000 - t0
                return button, rt

        return None, None