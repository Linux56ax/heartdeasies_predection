
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

#load dataset
data_path = "dataset.csv"   # Change if needed
df = pd.read_csv(data_path)
print("Dataset loaded:", df.shape)
print(df.head())

#data cleaning and preprocessing
df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

#map column names to consistent ones
rename_map = {
    "chest_pain_type": "chest_pain_type",
    "resting_bp_s": "resting_bp_s",
    "cholesterol": "cholesterol",
    "fasting_blood_sugar": "fasting_blood_sugar",
    "resting_ecg": "resting_ecg",
    "max_heart_rate": "max_heart_rate",
    "exercise_angina": "exercise_angina",
    "oldpeak": "oldpeak",
    "st_slope": "st_slope",
    "target": "target",
}
df = df.rename(columns=rename_map)

#indentify numeric and categorical columns
numeric_cols = ["age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak"]
categorical_cols = ["sex", "chest_pain_type", "fasting_blood_sugar",
                    "resting_ecg", "exercise_angina", "st_slope"]

#handle missing values
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

#dataset spliting 
X = df.drop(columns=["target"])
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

#pre processing pipeline
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols)
], remainder='passthrough')

#model tarining and evaluation
model= RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

results = {}
best_model = None
best_f1 = 0

#pipeline
pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", model)
])

#train
pipe.fit(X_train, y_train)

#predict
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

#evaluate
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print("\nRandomForest Results")
print(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  ROC_AUC={roc:.3f}")

#save the trained model
joblib.dump(pipe, "heartdeasies_predict.joblib")
print("\nModel saved as heartdeasies_predict.joblib")

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix - RandomForest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, int(val), ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.show()


