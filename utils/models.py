“””
Model training utilities — Ensemble (Voting) & Neural Network (Keras MLP)
“””
import numpy as np
import os, pickle, warnings
warnings.filterwarnings(“ignore”)

# ── sklearn ──────────────────────────────────

from sklearn.ensemble import (
RandomForestClassifier,
GradientBoostingClassifier,
VotingClassifier,
AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score, precision_score, recall_score,
f1_score, roc_auc_score, confusion_matrix,
classification_report,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(**file**)), “models”)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────

# ENSEMBLE

# ─────────────────────────────────────────────

def build_ensemble():
“””
Soft-voting ensemble of 4 base learners:
1. Random Forest
2. Gradient Boosting
3. AdaBoost
4. Logistic Regression
SVM is used as a 5th hard-voter (no predict_proba by default).
“””
rf  = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
lr  = LogisticRegression(max_iter=1000, random_state=42)
svm = SVC(kernel=“rbf”, probability=True, random_state=42)

```
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("ada", ada), ("lr", lr), ("svm", svm)],
    voting="soft",
)
return ensemble
```

def train_ensemble(X_train, y_train, X_test, y_test, name=“heart”):
model = build_ensemble()
model.fit(X_train, y_train)

```
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = _compute_metrics(y_test, y_pred, y_prob)

path = os.path.join(MODEL_DIR, f"ensemble_{name}.pkl")
with open(path, "wb") as f:
    pickle.dump(model, f)

return model, metrics
```

def load_ensemble(name=“heart”):
path = os.path.join(MODEL_DIR, f”ensemble_{name}.pkl”)
if os.path.exists(path):
with open(path, “rb”) as f:
return pickle.load(f)
return None

# ─────────────────────────────────────────────

# NEURAL NETWORK (Keras MLP)

# ─────────────────────────────────────────────

def build_nn(input_dim, hidden_layers=(128, 64, 32), dropout=0.3):
“””
MLP with BatchNorm + Dropout designed for tabular health data.
Architecture:
Input → [Dense→BN→ReLU→Dropout] × n_layers → Dense(1, sigmoid)
“””
try:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

```
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
except ImportError:
    return None
```

def train_nn(X_train, y_train, X_test, y_test, name=“heart”):
try:
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

```
    input_dim = X_train.shape[1]
    model = build_nn(input_dim)
    if model is None:
        return None, {}

    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0,
    )

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    metrics = _compute_metrics(y_test, y_pred, y_pred_prob)
    metrics["history"] = {
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "accuracy": history.history["accuracy"],
        "val_accuracy": history.history["val_accuracy"],
    }

    model.save(os.path.join(MODEL_DIR, f"nn_{name}.keras"))
    return model, metrics
except ImportError:
    return None, {}
```

def load_nn(name=“heart”):
try:
import tensorflow as tf
path = os.path.join(MODEL_DIR, f”nn_{name}.keras”)
if os.path.exists(path):
return tf.keras.models.load_model(path)
except Exception:
pass
return None

# ─────────────────────────────────────────────

# HELPERS

# ─────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, y_prob):
return {
“accuracy”:  round(accuracy_score(y_true, y_pred), 4),
“precision”: round(precision_score(y_true, y_pred, zero_division=0), 4),
“recall”:    round(recall_score(y_true, y_pred, zero_division=0), 4),
“f1”:        round(f1_score(y_true, y_pred, zero_division=0), 4),
“roc_auc”:   round(roc_auc_score(y_true, y_prob), 4),
“confusion_matrix”: confusion_matrix(y_true, y_pred).tolist(),
“report”: classification_report(y_true, y_pred, output_dict=True),
}

def evaluate_model(model, X_test, y_test, is_nn=False):
if is_nn:
y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)
else:
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
return _compute_metrics(y_test, y_pred, y_prob)
