import pandas as pd
from pathlib import Path

bids_root = Path("/Users/Barbu/Documents/Postdoc/Experiments/Pushmi/MEG/Data/Pilot/BIDS")

server = False
hard_drive = True
pc_id = "pc_id"

# if server:
#     path_root = "/neurospin/meg/meg_tmp/2024_ReplaySeq_Elyes/replayseq/2-Data/"
# else:
#     if hard_drive:
#         path_root = "/Volumes/CRAB/REPLAYSEQ/2-Data/"
#     else:
#         path_root = f"/volatile/home/{pc_id}/Documents/replayseq/2-Data/"

path_root = "/Users/Barbu/Documents/Postdoc/Experiments/Pushmi/MEG/Data/Pilot/"

study_name = "Pushmi"

bids_root = path_root + "BIDS"

task = "pushmi"

runs = 'all'

ch_types = ['meg']
data_type = 'meg' 
subjects = ['01']

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True

mf_ctc_fname = bids_root + "/system_calibration_files/ct_sparse.fif"
mf_cal_fname = bids_root + "/system_calibration_files/sss_cal_3176_20240123_2.dat"

# =============== Filtering parameters ==================

# -------------------------------------------------------
l_freq = 0.5
h_freq = 40  # TODO: DOUBLE CHECK THIS?
notch_freq = 50
raw_resample_sfreq = 250

# ============== Artifact correction ====================
"""spatial_filter = "ica"
ica_algorithm = 'fastica'
ica_n_components = 80
ica_max_iterations = 1000
random_state = 10
ica_l_freq = 1.0

# ==== ICA rejection settings (set to none, manual ICA) ====
ica_reject = None
"""

# epoching params
epochs_tmin = -0.2
epochs_tmax = 1
epochs_decim = 1
baseline = None

# Conditions / events to consider when epoching
conditions = ['initial_fixation', 
              'assignment_1_shape', 'assignment_1_label', 
              'inter_assignment_fixation',
              'assignment_2_shape', 'assignment_2_label', 
              'post_assignment_fixation', 
              'central_location_flash',
              'post_central_fixation', 
              'peripheral_location_flash', 
              'post_peripheral_fixation', 
              'outcome',
              'post_outcome_fixation', 
              'test_sentence_1', 'test_sentence_2', 'test_sentence_3', 'test_sentence_4', 'test_sentence_5',
              'response']

# reject = "autoreject_local"
reject = None
epochs_reject_by_annotation = False
n_jobs = 8