# Student Performance Prediction Using Machine Learning with Python

This repository contains an end-to-end, academic-grade machine learning project designed to predict student performance categories (**Poor, Average, Good, Excellent**) based on scholastic, behavioral, and demographic indicators. 

The project emphasizes modular code design, robust preprocessing (with zero data leakage), cross-validation, and multi-metric performance auditing.

---

## 📢 Synthetic Data Disclosure
> [!IMPORTANT]
> All data used in this project is **synthetically generated** and does not represent real student records. Anomalies (duplicates, missing fields, and out-of-domain values) have been programmatically injected solely to demonstrate and validate the robust data-cleaning pipelines.

---

## 🛠️ Setup & Installation

### 1. Clone/Initialize Repository
Create the directory structure and navigate to the project root:
```bash
cd student-performance-prediction
```

### 2. Install Dependencies
Ensure you have Python 3.12+ installed. Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Install Source package in Editable Mode
This registers the `src/` modules in your local path:
```bash
pip install -e .
```

---

## 📂 Project Architecture

```text
student-performance-prediction/
│
├── .gitignore                      # Ignores data, logs, models, and virtual environments
├── README.md                       # Comprehensive project report and usage guide
├── requirements.txt                # Fixed library versions
├── setup.py                        # Package builder script
│
├── data/
│   ├── raw/                        # Injected raw dataset with anomalies (student_data_raw.csv)
│   └── processed/                  # Cleaned, validated, and engineered dataset (student_data_processed.csv)
│
├── src/                            # Package source modules
│   ├── __init__.py
│   ├── data_generation.py          # Synthetic dataset simulation script
│   ├── features.py                 # Mathematical formulations of engineered features
│   ├── preprocessing.py            # Pipelines for validation, imputation, winsorization, and scaling
│   ├── models.py                   # Classifier definitions, GridSearchCV, and model serialization
│   ├── evaluation.py               # Plotting confusion matrices, ROC curves, and logging metrics
│   ├── test_prediction.py          # Script verifying inference on new student profiles
│   └── utils.py                    # Constants, path mappings, and standard loggers
│
├── models/                         # Serialized preprocessors and model estimators
│   ├── best_student_model.joblib   # Unified pipeline (Preprocessor + Best Classifier)
│   └── data_pipeline.joblib        # Preprocessing pipeline only
│
├── results/
│   └── reports/                    # Performance summaries (JSON) and evaluation plots (PNG)
│
└── app/
    └── app.py                      # Interactive Streamlit prediction dashboard
```

---

## 🔬 Methodology

### 1. Feature Engineering Math Logic
The following 4 composite features are engineered in `src/features.py` using domain-driven mathematical formulations:
*   **Study Intensity**: Combined measure of study duration and achievement efficacy.
    $$\text{Study Intensity} = \frac{\text{Daily Study Hours} \times \text{Assignment Score}}{100.0}$$
*   **Attendance Category**: Discrete categorical indicator representing risk.
    $$\text{Attendance Category} = \begin{cases} 
      \text{"Low"} & \text{if Attendance Percentage } < 75.0\% \\
      \text{"Moderate"} & \text{if } 75.0\% \le \text{Attendance Percentage } \le 90.0\% \\
      \text{"High"} & \text{if Attendance Percentage } > 90.0\% 
   \end{cases}$$
*   **Academic Consistency**: Deviation metric highlighting performance stability. Higher values imply more consistent outputs.
    $$\text{Academic Consistency} = 100.0 - |\text{Previous Semester Marks} - \text{Assignment Score}|$$
*   **Engagement Score**: Weighted behavior score representing overall participation (bounded between 0 and 100).
    $$\text{Engagement Score} = 0.4 \times \text{Attendance Percentage} + 0.4 \times \text{Class Participation} + 0.2 \times \text{Extracurricular Indicator}$$
    *(Where the Extracurricular Indicator = 100.0 if the student participates in extracurricular activities, otherwise 0.0)*

### 2. Preprocessing & Data Cleaning Pipeline
To prevent **data leakage**, all transformations are implemented inside an integrated Scikit-learn `ColumnTransformer` pipeline:
*   **Domain Validation (First Stage)**: Before transformations, values are screened against strict ranges:
    *   *Attendance, Assignment Score, Previous Semester Marks, Class Participation*: `[0.0, 100.0]`
    *   *Daily Study Hours*: `[0.0, 24.0]`
    *   *Sleep Hours*: `[4.0, 16.0]`
    *   *Out-of-domain anomalies are converted to `NaN`*.
*   **Missing Value Imputation**: Numerical variables are imputed using column `median`; categorical variables are imputed using `most_frequent` (mode).
*   **Outlier Winsorization**: Capping numerical columns at the 1st and 99th percentiles based solely on training set quantiles.
*   **Categorical Encoding**:
    *   *Nominal features* (`Gender`, `Internet Access`, `Extracurriculars`) are one-hot encoded with `drop='first'`.
    *   *Ordinal features* (`Family Income`, `Parent Education`, `Attendance Category`) are ordinal-encoded with custom-defined hierarchies and standardized.

---

## 📈 Model Comparison & Evaluation Results

All metrics below are generated by **actually executing** the models on the holdout test dataset (200 records, 80/20 train-test split):

| Model | Test Accuracy | Test Precision (Macro) | Test Recall (Macro) | Test F1-Score (Macro) | Test ROC-AUC (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Best)** | **0.6200** | **0.5854** | **0.4828** | **0.5091** | **0.8177** |
| Decision Tree | 0.5050 | 0.4238 | 0.3807 | 0.3905 | 0.7038 |
| Random Forest | 0.5850 | 0.4757 | 0.4048 | 0.4109 | 0.8076 |
| Support Vector Machine (SVM) | 0.5700 | 0.4843 | 0.4474 | 0.4599 | 0.7854 |
| K-Nearest Neighbors (KNN) | 0.5300 | 0.4584 | 0.3632 | 0.3661 | 0.6834 |

### Key Findings
*   **Logistic Regression** outperformed more complex tree-based classifiers. Since the target variable is mapped using a linear combination of normalized inputs, the linear baseline fits the underlying distribution best.
*   **One-vs-Rest ROC-AUC** scores are high (up to **0.8177**), indicating good overall class discrimination, while the macro F1-score indicates that classification performance varies across the individual performance categories.

---

## 🚀 Execution Guide

### Step 1: Generate Raw Synthetic Data
```bash
python src/data_generation.py
```
*Generates 1,000 base records, injects 20 duplicates, 3% missing values, and out-of-domain outliers, saving raw data to `data/raw/student_data_raw.csv`.*

### Step 2: Clean, Train, and Evaluate Models
```bash
python src/models.py
```
*Applies cleaning/preprocessing pipelines, executes GridSearchCV with Stratified 5-Fold Cross-Validation, compares metrics on the holdout test set, saves evaluation plots to `results/reports/`, and serializes the best model to `models/best_student_model.joblib`.*

### Step 3: Run Standalone Test Inference
```bash
python src/test_prediction.py
```
*Loads the serialized pipeline and runs inference on a new mock student profile, outputting class probabilities.*

### Step 4: Run Streamlit Interface
```bash
streamlit run app/app.py
```
*Launches an interactive browser window to adjust student inputs, view predictions, and inspect training plots.*
