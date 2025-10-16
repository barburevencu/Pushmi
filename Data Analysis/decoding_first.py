import mne
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from mne.decoding import Scaler, Vectorizer, cross_val_multiscore
from mne.decoding import LinearModel, SlidingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from matplotlib.animation import FuncAnimation, FFMpegWriter

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

epochs.metadata = (
    behavior
      .join(meta, rsuffix="_meta")
      .rename(columns={"trial_type": "task"})
      .assign(
      stimulus=lambda d: d["shape1"].fillna(d["label1"]),
      event_name=lambda d: d["event_name"].astype(str) + "_" + d["stimulus"].astype(str)
      ))

mapping = {v:(i+1) for i, v in enumerate(set(epochs.metadata['event_name']))}
labels = epochs.metadata['event_name'].map(mapping).to_list()

# clf = make_pipeline(
#     Scaler(epochs.info),  # Normalisation des données
#     Vectorizer(),  # Transforme les données 3D en 2D pour sklearn
#     LinearModel(LogisticRegression(solver='liblinear', C=1))  # Modèle de classification
# )

clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability = True))

# Création d'un estimateur glissant pour le décodage temporel
time_decoding = SlidingEstimator(clf, n_jobs=1, scoring='accuracy', verbose=True)

# Exécution de la validation croisée
print("\n--- Exécution de la validation croisée ---")
epochs.decimate(2)
X = epochs.pick_types(meg='mag').get_data()
y = np.array(labels)
scores = cross_val_multiscore(time_decoding, X , labels, cv=5, n_jobs=-1)

mean_scores = np.mean(scores, axis=0)
std_scores = np.std(scores, axis=0)

n_classes = len(mapping.keys())
# Visualisation des résultats de décodage
print("\n--- Visualisation des résultats de décodage ---")
# Création de la figure
fig, ax = plt.subplots(figsize=(10, 6))

# Tracé de la courbe de performance du décodage
times = epochs.times
ax.plot(times, mean_scores, label='Decoding score', color='blue')
ax.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                color='blue', alpha=0.2, label='Standard deviation')

# Ajout du niveau de chance
chance_level = 1.0 / n_classes
ax.axhline(chance_level, color='k', linestyle='--', label=f'Chance level ({chance_level:.2f})')

# Formatage du graphique
ax.set_xlabel('Time (s)')
ax.set_ylabel('Decoding performance')
ax.set_title('Decoding')
ax.legend()
ax.set_ylim([0, 1])
ax.axvline(0, color='k', linestyle='-', alpha=0.2)

plt.tight_layout()
plt.show()

# ==================
# Time-resolved confusion matrix video
# ==================
print("\n--- Computing time-resolved confusion matrices ---")
# Use only magnetometers already picked above; X has shape (n_trials, n_sensors, n_times)
n_trials, n_sensors, n_times = X.shape

# Ensure deterministic, stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Establish a stable class order [1..n_classes] matching our 'labels'
class_order = np.array(sorted(set(y)))
assert len(class_order) == n_classes, f"Expected {n_classes} classes, got {len(class_order)}"

# Allocate array to store confusion matrices (time, true_class, pred_class)
cms = np.zeros((n_times, n_classes, n_classes), dtype=float)

# For each time point, get cross-validated predictions and build a confusion matrix
for ti in range(n_times):
    Xt = X[:, :, ti]  # shape (n_trials, n_sensors)
    # cross-validated predictions at this time point
    y_pred = cross_val_predict(clf, Xt, y, cv=cv, n_jobs=-1, method='predict')
    cm = confusion_matrix(y, y_pred, labels=class_order)

    # Normalize rows to sum to 1 (per true class) to reflect accuracies
    with np.errstate(invalid='ignore', divide='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    cms[ti] = cm_norm

print("--- Confusion matrices computed ---")

# ------------------
# Create and save an MP4 video
# ------------------
OUT_VIDEO = Path("confusion_over_time.mp4")

fig_cm, ax_cm = plt.subplots(figsize=(8, 7))
im = ax_cm.imshow(cms[0], vmin=0.0, vmax=1.0, origin='upper', interpolation='nearest', aspect='auto')
cbar = plt.colorbar(im, ax=ax_cm)
cbar.set_label("Proportion (per true class)")

ax_cm.set_xlabel("Predicted class")
ax_cm.set_ylabel("True class")
ax_cm.set_title(f"Time {times[0]:.3f} s")
ax_cm.set_xticks(np.arange(n_classes))
ax_cm.set_yticks(np.arange(n_classes))
ax_cm.set_xticklabels(class_order)
ax_cm.set_yticklabels(class_order)
plt.tight_layout()

def update(frame_idx):
    im.set_data(cms[frame_idx])
    ax_cm.set_title(f"Time {times[frame_idx]:.3f} s")
    return (im,)

anim = FuncAnimation(fig_cm, update, frames=n_times, interval=1, blit=True)

try:
    writer = FFMpegWriter(fps=30, metadata={'artist': 'decoding_first.py'})
    anim.save(str(OUT_VIDEO), writer=writer)
    print(f"Saved video to: {OUT_VIDEO}")
except Exception as e:
    print(f"FFmpeg not available or failed with error: {e}")
    print("As a fallback, saving as GIF (this may be large).")
    anim.save("confusion_over_time.gif", dpi=100)
    print("Saved fallback GIF: confusion_over_time.gif")