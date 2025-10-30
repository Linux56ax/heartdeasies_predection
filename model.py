
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import joblib
import matplotlib.pyplot as plt

# 1. Load dataset
data_path = "/mnt/data/dataset.csv"
df = pd.read_csv(data_path)

# show a quick preview
print("Dataset loaded from:", data_path)
print("Shape:", df.shape)
display_df = df.copy()
# Display first rows using the provided helper for DataFrames if available
try:
    # caas_jupyter_tools.display_dataframe_to_user is available in the environment used by python_user_visible
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Dataset preview", df.head(10))
except Exception:
    print(df.head(10))

# 2. Basic inspection
print("\nColumns and dtypes:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nBasic statistics:")
display(df.describe(include='all').T)

# Normalize column names (remove spaces / lowercase) to avoid key errors
df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

# Expected columns (based on PDF). We'll try to map common names if necessary.
expected_cols = {
    "age": "age",
    "sex": "sex",
    "chest_pain_type": None,
    "chestpain": None,
    "cp": None,
    "resting_bp_s": None,
    "resting_bp": None,
    "resting_bp_s_in_mm_hg": None,
    "cholesterol": "cholesterol",
    "serum_cholesterol": None,
    "fasting_blood_sugar": None,
    "fasting_blood": None,
    "resting_ecg": None,
    "resting_ecg_results": None,
    "max_heart_rate": None,
    "maximum_heart_rate_achieved": None,
    "exercise_angina": None,
    "exercise_angina_yn": None,
    "oldpeak": "oldpeak",
    "st_slope": None,
    "slope": None,
    "class": None,
    "target": None
}

# try to infer mappings from existing columns
col_map = {}
for c in df.columns:
    if c in expected_cols and expected_cols[c]:
        col_map[c] = expected_cols[c]
    # variations
    if "cp" in c or "chest" in c:
        col_map[c] = "chest_pain_type"
    if "rest" in c and "bp" in c:
        col_map[c] = "resting_bp_s"
    if "chol" in c:
        col_map[c] = "cholesterol"
    if "fast" in c and "sugar" in c:
        col_map[c] = "fasting_blood_sugar"
    if "ecg" in c:
        col_map[c] = "resting_ecg"
    if "max" in c and ("heart" in c or "hr" in c):
        col_map[c] = "max_heart_rate"
    if "exercise" in c and "ang" in c:
        col_map[c] = "exercise_angina"
    if "old" in c and "peak" in c:
        col_map[c] = "oldpeak"
    if "slope" in c or ("st" in c and "slope" in c):
        col_map[c] = "st_slope"
    if c in ("class", "target", "heartdisease", "has_disease", "heart_disease"):
        col_map[c] = "target"

# Apply mapping
df = df.rename(columns=col_map)
print("\nColumns after mapping:")
print(df.columns.tolist())

# Verify target exists
if "target" not in df.columns:
    # try last column as target
    df = df.rename(columns={df.columns[-1]: "target"})
    print("\nRenamed last column to 'target' - new columns:")
    print(df.columns.tolist())

# 3. Handle missing values: impute numeric with median, categorical with most_frequent
# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# exclude target from features
if "target" in numeric_cols:
    numeric_cols.remove("target")
# for categorical, consider columns with few unique values or object dtype
categorical_cols = [c for c in df.columns if c not in numeric_cols + ["target"]]

# Also treat some nominal-coded columns as categorical
for col in ["sex", "chest_pain_type", "fasting_blood_sugar", "resting_ecg", "exercise_angina", "st_slope"]:
    if col in df.columns and col not in categorical_cols:
        categorical_cols.append(col)
        if col in numeric_cols:
            numeric_cols.remove(col)

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Fill missing values if any
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Ensure categorical columns are integer type
for c in categorical_cols:
    try:
        df[c] = df[c].astype(int)
    except:
        df[c] = pd.Categorical(df[c]).codes

# 4. Train-test split
X = df.drop(columns=["target"])
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

print("\nTrain/test sizes:", X_train.shape, X_test.shape, y_train.value_counts(normalize=True).to_dict())

# 5. Build preprocessing + model pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# For categorical features we will leave them as-is (they're coded integers).
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols)
    ],
    remainder='passthrough'  # leave categorical columns untouched
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = {}
trained_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps['clf'], "predict_proba") else pipe.decision_function(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}
    trained_pipelines[name] = pipe
    print(f"\n{name} metrics:")
    print(results[name])

# Show results in a dataframe
results_df = pd.DataFrame(results).T.sort_values(by="f1", ascending=False)
display(results_df)

# 6. Choose best model by F1 (or ROC if tie)
best_name = results_df.index[0]
best_pipe = trained_pipelines[best_name]
print("\nBest model by F1 score:", best_name)
print(results_df.loc[best_name])

# 7. Save best model and test predictions
out_model_path = "/mnt/data/best_heart_model.joblib"
joblib.dump(best_pipe, out_model_path)

# Save test predictions
y_pred_best = best_pipe.predict(X_test)
try:
    y_proba_best = best_pipe.predict_proba(X_test)[:,1]
except:
    y_proba_best = best_pipe.decision_function(X_test)
pred_df = X_test.copy()
pred_df["y_true"] = y_test.values
pred_df["y_pred"] = y_pred_best
pred_df["y_proba"] = y_proba_best
out_pred_path = "/mnt/data/test_predictions.csv"
pred_df.to_csv(out_pred_path, index=False)

print("\nSaved best model to:", out_model_path)
print("Saved test predictions to:", out_pred_path)

# 8. Confusion matrix and ROC curve for best model
cm = confusion_matrix(y_test, y_pred_best)
fpr, tpr, _ = roc_curve(y_test, y_proba_best)

# Plot confusion matrix
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, int(val), ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.show()

# Plot ROC
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle='--')
plt.title(f"ROC Curve - {best_name}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the final selected model metrics
print("\nFinal selected model and metrics:")
print(best_name, results[best_name])

# Provide paths for download
print("\nDownloadable files:")
print("Best model:", out_model_path)
print("Test predictions CSV:", out_pred_path)
