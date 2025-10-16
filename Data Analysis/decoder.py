import mne
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from mne.decoding import Scaler, Vectorizer, cross_val_multiscore
from mne.decoding import LinearModel, SlidingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

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

# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=ncv, shuffle=True, random_state=42)
# # Loop through the stratified splits
# y_true_cvs = []
# y_preds = []
# for train_index, test_index in skf.split(X, y_true):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y_true[train_index], y_true[test_index]
#     decoder.fit(X_train,y_train)
#     y_preds.append(decoder.predict(X_test))
#     y_true_cvs.append(y_true[test_index])

    # clf.predict(X)

# ------------------
# Paths & I/O
# ------------------
LOCALIZER_EPO = Path("/Users/Barbu/Documents/Postdoc/Experiments/Pushmi/MEG/Data/Pilot/BIDS/derivatives/mne-bids-pipeline/sub-01/meg/sub-01_task-localizer_epo.fif")
BEHAVIOR_XPD = Path("/Users/Barbu/Documents/Postdoc/Experiments/Pushmi/MEG/Data/Pilot/Behavior/sub-01/data.xpd")

epochs = mne.read_epochs(LOCALIZER_EPO)
meta = epochs.metadata.reset_index(drop=True)

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

clf = make_pipeline(
    Scaler(epochs.info),  # Normalisation des données
    Vectorizer(),  # Transforme les données 3D en 2D pour sklearn
    LinearModel(LogisticRegression(solver='liblinear', C=1))  # Modèle de classification
)

# Création d'un estimateur glissant pour le décodage temporel
time_decoding = SlidingEstimator(clf, n_jobs=1, scoring='accuracy', verbose=True)

# Exécution de la validation croisée
print("\n--- Exécution de la validation croisée ---")
epochs.decimate(2)
X = epochs.pick_types(meg='mag').get_data()
y = epochs.events[:,2]
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