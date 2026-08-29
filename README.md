```markdown
# ❤️ Heart Disease Prediction System (ML & Streamlit Web App)

---

## ⚠️ Medical Disclaimer

> **IMPORTANT:** This application is built strictly for educational, research, and demonstration purposes. It is **not** a certified medical diagnostic tool and **must not be used to replace professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider or physician for any questions regarding a medical condition or clinical evaluation.

---

An end-to-end Machine Learning web application that predicts patient heart disease risk based on clinical parameters. Built using Python, Scikit-learn, and Streamlit, this system integrates standard feature scaling and a Logistic Regression classifier trained on the Cleveland Heart Disease dataset.

---

## 📌 Project Overview
Heart disease is one of the primary causes of morbidity globally. This project delivers a clinical risk assessment tool using supervised machine learning. 

The application accepts 13 key physiological metrics entered by a user through an interactive web interface, processes the inputs using a fitted `StandardScaler` pipeline, and provides real-time binary risk predictions (**Low Risk** vs. **High Risk**).

---

## 🛠️ Tech Stack & Key Libraries
* **Frontend / Web Framework:** Streamlit
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Model Serialization:** Pickle

---

## 🎯 Key Features & System Architecture

* **Interactive Web Interface:** User-friendly UI built with Streamlit allowing custom inputs for all 13 medical parameters.
* **On-the-Fly Feature Scaling:** Preprocesses user inputs using `StandardScaler` fitted on training feature distributions to prevent model prediction bias.
* **Trained ML Engine:** Employs a fitted `LogisticRegression` classification model trained on the UCI Cleveland dataset.
* **Instant Risk Assessment:** Generates color-coded visual alerts indicating low risk or high risk of heart disease.

---

## 📊 Dataset & Medical Parameters

The model evaluates patient profiles using the following 13 clinical attributes:

| Feature Name | Description | Value Type / Encoding |
| :--- | :--- | :--- |
| **`age`** | Patient Age | Numerical (Years) |
| **`sex`** | Patient Gender | `0 = Female`, `1 = Male` |
| **`cp`** | Chest Pain Type | `0: Typical`, `1: Atypical`, `2: Non-anginal`, `3: Asymptomatic` |
| **`trestbps`** | Resting Blood Pressure | Numerical (mm Hg) |
| **`chol`** | Serum Cholesterol | Numerical (mg/dl) |
| **`fbs`** | Fasting Blood Sugar > 120 mg/dl | `0 = False`, `1 = True` |
| **`restecg`** | Resting ECG Results | `0: Normal`, `1: ST-T Wave Abnormality`, `2: LV Hypertrophy` |
| **`thalach`** | Maximum Heart Rate Achieved | Numerical (bpm) |
| **`exang`** | Exercise-Induced Angina | `0 = No`, `1 = Yes` |
| **`oldpeak`** | ST Depression Induced by Exercise | Numerical (mm) |
| **`slope`** | Slope of Peak Exercise ST Segment | `0: Upsloping`, `1: Flat`, `2: Downsloping` |
| **`ca`** | Major Vessels Colored by Fluoroscopy | `0 – 3` |
| **`thal`** | Thalassemia | `0: Normal`, `1: Fixed Defect`, `2: Reversible Defect` |

---

## ⚙️ Model Target & Logic Mapping

* **`0` = Low Risk** (No presence of heart disease)
* **`1` = High Risk** (Presence of heart disease detected)

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure Python 3.8+ is installed on your machine.

### 2. Clone the Repository
```bash
git clone [https://github.com/supriya20242007/codealphadiseasepredictor.git](https://github.com/supriya20242007/codealphadiseasepredictor.git)
cd codealphadiseasepredictor

```

### 3. Install Dependencies

```bash
python -m pip install streamlit pandas numpy scikit-learn

```

### 4. Launch the Streamlit App

```bash
python -m streamlit run app.py

```

Open `http://localhost:8501` in your web browser.

---

## 📁 Repository Structure

```text
codealphadiseasepredictor/
├── app.py                         # Streamlit Web Application Interface
├── Heart_Disease_Prediction.ipynb # Model Training, EDA & Evaluation Notebook
├── heart_cleveland_upload.csv     # Dataset for Feature Scaling Reference
├── heart_disease_model.pkl        # Serialized Machine Learning Model
└── README.md                      # Project Documentation

```

```

```