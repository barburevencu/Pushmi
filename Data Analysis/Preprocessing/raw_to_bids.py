import mne_bids
import mne
import numpy as np

def prepare_data_for_mne_bids_pipeline(subject='01', 
                                       base_path="../Data/Pilot",
                                       task_name='pushmi', 
                                       cal_filename='sss_cal_3176_20240123_2.dat', 
                                       ct_filename='ct_sparse.fif',
                                       run_names=[f"{i:02}" for i in range(1, 13)]):
    """
    Prepare and convert MEG data to BIDS format for MNE-BIDS pipeline processing.
    """

    original_data_path = base_path + "/Raw/"
    root = base_path + '/BIDS/'

    for run in run_names:
        print(f"--- saving in bids format run {run} ---")
        data_path = original_data_path + f"sub-{subject}" + f'/run-{run}.fif'
        raw = mne.io.read_raw_fif(data_path, allow_maxshield=True, preload=False)

        events, event_ids = extract_events_and_event_IDs(raw)
        # Create the BIDS path
        bids_path = mne_bids.BIDSPath(subject=subject, task=task_name, run=run, datatype='meg', root=root)
        # Write the raw data
        mne_bids.write_raw_bids(raw, bids_path=bids_path, allow_preload=True, format='FIF',events=events,
                       event_id=event_ids, overwrite=True)


        # Write MEG calibration and crosstalk files
        cal_fname = root + f'/system_calibration_files/{cal_filename}'
        ct_fname = root + f'/system_calibration_files/{ct_filename}'
        mne_bids.write_meg_calibration(calibration=cal_fname, bids_path=bids_path)
        mne_bids.write_meg_crosstalk(fname=ct_fname, bids_path=bids_path)


def extract_events_and_event_IDs(raw, trigger_labels=None):
    """
    Extracts specific events and their IDs from raw MEG/EEG data using MNE,
    including BAD annotations if present.
    """
    trigger_labels = {
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

    # Extract regular stimulus events
    events = mne.find_events(raw, min_duration=0.01, mask_type="not_and",
                             mask=2 ** 6 + 2 ** 7 + 2 ** 8 + 2 ** 9 + 2 ** 10 + 2 ** 11 + 2 ** 12 + 2 ** 13 + 2 ** 14 + 2 ** 15)

    # filter the events and event_dict such that we keep only the ones in trigger labels
    events = np.vstack([ev for ev in events if ev[2] in list(trigger_labels.values())])
    trigger_labels = {k: v for k, v in trigger_labels.items() if v in np.unique(events[:, 2])}

    return events, trigger_labels

# prepare_data_for_mne_bids_pipeline(subject='01', base_path="../../Data/Pilot",
#                                    task_name='pushmi',
#                                    run_names=[f"{i:02}" for i in range(1, 13)])

subject = '01'
for run in [str(f"{i:02}") for i in range(8, 9)]:
    bids_root = "/Users/Barbu/Documents/Postdoc/Experiments/Pushmi/MEG/Data/Pilot/BIDS"
    bids_path = mne_bids.BIDSPath(subject=subject, datatype="meg", root=bids_root,task="pushmi",run=run)
    mne_bids.inspect_dataset(bids_path)