import pylink
import time
from screeninfo import get_monitors

class EyeLinkWrapper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.block_counter = 0
        self.el_tracker = pylink.EyeLink('100.1.1.1')
        self.screen_size = self.get_screen_size()
        self.current_edf_file_name = self.get_next_edf_file_name()

    def get_next_edf_file_name(self):
        """Generate the next EDF file name based on the base name and block counter."""
        return f"{self.cfg['edf_file_base_name']}r{self.block_counter:02}.edf"

    def get_screen_size(self):
        """Get monitor size and print monitor information."""
        monitors = get_monitors()
        for i, monitor in enumerate(monitors):
            print(f"Monitor {i}: {monitor}")

        if monitors:
            monitor = monitors[0]  # Use the first monitor
            size = (monitor.width, monitor.height)
            print(f"Chosen monitor: {monitor}")
            return size
        else:
            raise RuntimeError("No monitors found")

    def initialize(self):
        """Initialize the EyeLink connection and start the first block."""
        self.start_new_block()

    def start_new_block(self):
        """Start a new block by opening a new EDF file and configuring the tracker."""
        if self.block_counter > 0:
            # Close the current EDF file before starting a new one
            self.el_tracker.closeDataFile()
            self.el_tracker.receiveDataFile(self.current_edf_file_name, self.current_edf_file_name)
        
        self.current_edf_file_name = self.get_next_edf_file_name()
        self.block_counter += 1

        # Set the EyeLink to offline mode
        self.el_tracker.setOfflineMode()

        # Initialize EyeLink graphics
        screen_width = self.screen_size[0]
        screen_height = self.screen_size[1]
        #self.el_tracker.openGraphics((screen_width, screen_height))

        # Open a new data file
        try:
            self.el_tracker.openDataFile(self.current_edf_file_name)
        except RuntimeError as e:
            print(f"Error opening data file '{self.current_edf_file_name}': {e}")

        self.el_tracker.sendCommand("screen_pixel_coords =  0 0 %d %d" % (screen_width-1, screen_height-1))
        self.el_tracker.sendMessage("DISPLAY_COORDS  0 0 %d %d" % (screen_width-1, screen_height-1))

        # Initialize tracker software version
        tracker_software_ver = 0

        # Get EYELINK version and software version
        eyelink_ver = self.el_tracker.getTrackerVersion()
        if eyelink_ver == 3:
            tvstr = self.el_tracker.getTrackerVersionString()
            vindex = tvstr.find("EYELINK CL")
            tracker_software_ver = int(float(tvstr[vindex + len("EYELINK CL"):].strip()))

        # Set specific settings based on the tracker version
        if eyelink_ver >= 2:
            self.el_tracker.sendCommand("select_parser_configuration 0")
        if eyelink_ver == 2:
            # Turn off SceneLink camera
            self.el_tracker.sendCommand("scene_camera_gazemap = NO")
        else:
            # Configure saccade thresholds for versions other than 2
            self.el_tracker.sendCommand("saccade_velocity_threshold = 35")
            self.el_tracker.sendCommand("saccade_acceleration_threshold = 9500")

        # Set EDF file contents
        self.el_tracker.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON")
        if tracker_software_ver >= 4:
            pylink.getEYELINK().sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,HTARGET")
        else:
            pylink.getEYELINK().sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS")

        # Set link data. This specifies which data is sent through the link and thus be used in gaze contingent displays
        self.el_tracker.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON")
        if tracker_software_ver >= 4:
            self.el_tracker.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,HTARGET")
        else:
            self.el_tracker.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS")

        # Configure button function to accept fixation during drift
        self.el_tracker.sendCommand("button_function %d 'accept_target_fixation'" % (5))

        # Make sure that we are connected to the eyelink before we start further communication
        if not self.el_tracker.isConnected():
            raise RuntimeError("Failed to connect to the eyetracker")

    def calibrate(self):
        """Calibrate the EyeLink eye-tracker."""
        # Perform tracker setup to start calibration
        try:
            pylink.openGraphics()

            # Configure calibration settings
            self.el_tracker.sendCommand("calibration_type = HV9")
            self.el_tracker.sendCommand("generate_default_targets = YES")
            self.el_tracker.sendCommand("calibration_area_proportion = 0.7 0.65")
            self.el_tracker.sendCommand("validation_area_proportion = 0.68 0.63")
            pylink.setCalibrationColors(self.cfg['foreground_color'],self.cfg['background_color']) # White target, gray background
            pylink.setCalibrationSounds("", "", "") # No sounds for calibration
            pylink.setDriftCorrectSounds("", "off", "off") # No sounds for drift correction

            self.el_tracker.doTrackerSetup()

            pylink.closeGraphics()

        except Exception as e:
            print(f"Error during calibration: {e}")

    def start_recording(self):
        """Start recording with EyeLink."""
        try:
            # Attempt to start recording
            error = self.el_tracker.startRecording(1, 1, 1, 1)  # 0 if successful
            if error:
                raise RuntimeError("Could not start recording.")
            print("Eye-tracker recording started...")

            # Enable real-time mode with high priority
            #self.el_tracker.beginRealTimeMode(100)
            #print("Real-time mode enabled.")

            # Wait for the tracker to start sending data
            if not self.el_tracker.waitForBlockStart(1000, 1, 0):
                # Stop recording if no data received within the timeout
                self.stop_recording()
                self.el_tracker.sendMessage("TRIAL ERROR")
                raise RuntimeError("No link samples received.")

            # Close EyeLink graphics if necessary
            pylink.closeGraphics()

        except (RuntimeError, Exception) as e:
            # Handle all exceptions
            print(f"Error: {e}")
            self.stop_recording()
            self.el_tracker.sendMessage("TRIAL ERROR")

    def stop_recording(self):
        """Stop recording with EyeLink."""
        #self.el_tracker.endRealTimeMode()
        time.sleep(0.1)
        self.el_tracker.stopRecording()

    def close(self):
        """Close the EyeLink connection and transfer the data file."""
        self.el_tracker.setOfflineMode()                         
        time.sleep(0.5) 
        self.el_tracker.closeDataFile()
        self.el_tracker.receiveDataFile(self.current_edf_file_name, self.current_edf_file_name)
        self.el_tracker.close()

    def send_message(self, message):
        """Send a message to the EyeLink data file."""
        self.el_tracker.sendMessage(message)