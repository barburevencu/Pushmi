from pathlib import Path
import pandas as pd
import mne

# ------------------
# Paths & I/O
# ------------------
LOCALIZER_EPO = Path("/Users/Barbu/Documents/Postdoc/Experiments/Pushmi/MEG/Data/Pilot/BIDS/derivatives/mne-bids-pipeline/sub-01/meg/sub-01_task-localizer_epo.fif")
BEHAVIOR_XPD = Path("/Users/Barbu/Documents/Postdoc/Experiments/Pushmi/MEG/Data/Pilot/Behavior/sub-01/data.xpd")

epochs = mne.read_epochs(LOCALIZER_EPO)
meta = epochs.metadata.reset_index(drop=True)

# ------------------
# Behavioral dataset
# ------------------
behavior = (
    pd.read_csv(BEHAVIOR_XPD, comment="#")
      .query("trial_type == 'localizer' and trial_block != 0")
      .assign(
          subject_id = 1,
          trial_number = lambda d: d.trial_number - 64
      )
      .loc[:, [
          "subject_id", "trial_type", "trial_block", "trial_number",
          "shape1", "label1", "correct_key",
          "participant_response", "reaction_time", "participant_correct"
      ]]
      .reset_index(drop=True)
)

# ------------------
# Align & merge by row index
# ------------------
assert len(behavior) == len(meta), f"Row mismatch: behavior={len(behavior)} vs meta={len(meta)}"

merged = (
    behavior
      .join(meta, rsuffix="_meta")
      .rename(columns={"trial_type": "task"})
      .assign(stimulus=lambda d: d["shape1"].fillna(d["label1"]))
)

epochs.metadata = merged