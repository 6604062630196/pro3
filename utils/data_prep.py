import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(**file**)), “data”)

# ─────────────────────────────────────────────

# HEART DISEASE

# ─────────────────────────────────────────────

def load_heart_raw():
path = os.path.join(DATA_DIR, “heart_disease.csv”)
return pd.read_csv(path)

def prepare_heart_data():
df = load_heart_raw()

```
# 1. Missing value report
missing_before = df.isnull().sum()

# 2. Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# 3. Remove duplicates
df.drop_duplicates(inplace=True)

# 4. Outlier capping (IQR) for trestbps and chol
for col in ["trestbps", "chol", "thalach"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

# 5. Feature / target split
X = df.drop("target", axis=1)
y = df["target"]

# 6. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

return {
    "df_raw": load_heart_raw(),
    "df_clean": df,
    "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test,
    "scaler": scaler,
    "feature_names": list(X.columns),
    "missing_before": missing_before,
}
```

# ─────────────────────────────────────────────

# DIABETES

# ─────────────────────────────────────────────

def load_diabetes_raw():
path = os.path.join(DATA_DIR, “diabetes.csv”)
return pd.read_csv(path)

def prepare_diabetes_data():
df = load_diabetes_raw()

```
missing_before = df.isnull().sum()

# Zero-values in certain columns should be treated as missing
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_as_missing:
    df[col] = df[col].replace(0, np.nan)

# Fill with median
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

df.drop_duplicates(inplace=True)

# Outlier capping
for col in ["Glucose", "BMI", "Insulin"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

return {
    "df_raw": load_diabetes_raw(),
    "df_clean": df,
    "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test,
    "scaler": scaler,
    "feature_names": list(X.columns),
    "missing_before": missing_before,
}
```
