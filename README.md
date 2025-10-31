Here’s a clean, professional **README.md** for your Heart Disease Detection project 👇

---
🩺 Heart Disease Detection using Machine Learning

📘 Overview

This project predicts whether a patient is likely to have **heart disease** based on their medical attributes such as age, blood pressure, cholesterol level, and more.
It uses **Random Forest Classifier** — a powerful and interpretable machine learning model — to perform binary classification.

---

🧠 Objective

To build an ML model that accurately classifies patients as:

* `0` → Normal (No Heart Disease)
* `1` → Heart Disease

---

🧾 Dataset

The dataset (`dataset.csv`) contains patient data with the following attributes:

| Feature             | Description                       | Type    |
| ------------------- | --------------------------------- | ------- |
| age                 | Age in years                      | Numeric |
| sex                 | 0 = Female, 1 = Male              | Binary  |
| chest_pain_type     | 1–4 (Types of chest pain)         | Nominal |
| resting_bp_s        | Resting blood pressure (mm Hg)    | Numeric |
| cholesterol         | Serum cholesterol (mg/dL)         | Numeric |
| fasting_blood_sugar | 1 if >120 mg/dL else 0            | Binary  |
| resting_ecg         | ECG results (0–2)                 | Nominal |
| max_heart_rate      | Maximum heart rate achieved       | Numeric |
| exercise_angina     | 1 = Yes, 0 = No                   | Binary  |
| oldpeak             | ST depression induced by exercise | Numeric |
| st_slope            | 1 = Up, 2 = Flat, 3 = Down        | Nominal |
| target              | 0 = Normal, 1 = Heart Disease     | Binary  |

---

⚙️ Steps Performed

1. **Data Preprocessing**

   * Cleaned column names
   * Handled missing values using median/mode imputation
   * Split dataset into training and testing (80–20)
   * Scaled numeric features

2. **Model Building**

   * Trained a **Random Forest Classifier**
   * Used balanced class weights to handle class imbalance

3. **Evaluation Metrics**

   * Accuracy
   * Precision
   * Recall
   * F1 Score
   * ROC-AUC

4. **Visualization**

   * Confusion Matrix
   * ROC Curve

---

🧩 Technologies Used

* **Python 3.9+**
* **pandas**, **numpy**
* **scikit-learn**
* **matplotlib**
* **joblib**

---

### 🚀 How to Run

#### 1. Clone this repository

```bash
git clone https://github.com/<your-username>/heart-disease-detector.git
cd heart-disease-detector
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run the model

```bash
python detect_heart_disease.py
```

#### 4. Output files

* `best_heart_model.joblib` — Trained Random Forest model
* `confusion_matrix` and `roc_curve` plots — Saved/displayed after training

---

### 📊 Example Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.92  |
| Precision | 0.93  |
| Recall    | 0.93  |
| F1 Score  | 0.93  |
| ROC-AUC   | 0.97  |

---

### 💾 Model Saving

The trained model is saved using `joblib`:

```python
joblib.dump(pipe, "best_heart_model.joblib")
```

You can later load it for predictions:

```python
model = joblib.load("best_heart_model.joblib")
pred = model.predict(new_data)
```

---

### 🧬 Future Improvements

* Integrate Flask API for real-time prediction
* Add feature importance analysis (SHAP or permutation importance)
* Deploy using Streamlit or FastAPI

---

### 👨‍💻 Author
GitHub: [https://github.com/Linux56ax](https://github.com/Linux56ax)
