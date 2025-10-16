import mne
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

from mne.decoding import Vectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Read MEG data and metadata
ROOT = Path(__file__).resolve().parent

LOCALIZER_EPO = (
    ROOT.parent / "Data" / "Pilot" / "BIDS" / "derivatives" /
    "mne-bids-pipeline" / "sub-01" / "meg" / "sub-01_task-localizer_epo.fif"
)

epochs = mne.read_epochs(LOCALIZER_EPO)
# Read labels
# (Ensure metadata rows match the number of kept epochs)
df = pd.read_csv("metadata.csv")

# Set X (data) and y (labels) for the classifier
# Use magnetometers and vectorize to 2D (n_trials, n_features) so OvO accepts the input
X = epochs.pick_types(meg='mag').get_data()
X = Vectorizer().fit_transform(X)  # pure reshape; no leakage concerns

# Labels from metadata
y = df["stimulus"].astype(str).values

# Sanity check
assert X.shape[0] == len(y), f"Number of epochs ({X.shape[0]}) != number of labels ({len(y)}). Ensure metadata aligns with kept epochs."

# Encode class labels to stable integer indices (ensures consistent column order across folds)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Define pipeline: standardize features within folds, then linear classifier
base_clf = LogisticRegression(solver="liblinear", max_iter=1000)

pipeline = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    base_clf,
)

# One-vs-One wrapper around the whole pipeline
ovo_clf = OneVsOneClassifier(pipeline)

# Cross-validated predictions
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Predict probabilities per trial (OvO returns pseudo-probabilities via pairwise coupling)
y_scores = cross_val_predict(ovo_clf, X, y_enc, cv=cv,
                             method="decision_function", n_jobs=-1)

# Predicted classes (indices) and back to original labels
y_pred_idx = np.argmax(y_scores, axis=1)
y_pred = le.inverse_transform(y_pred_idx)

# Confusion matrix (optionally normalized per true class)
labels = le.classes_
cm = confusion_matrix(y, y_pred, labels=labels)

# Optionally, get a row-normalized version for accuracies per class
cm_norm = cm / cm.sum(axis=1, keepdims=True)

# Plot
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, xticks_rotation=45)
plt.tight_layout()
plt.show()