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

task = "localizer"

runs = ['01', '02']

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
conditions = ['word', 'image']

import numpy as np
reject = "autoreject_local"

n_jobs = 8