import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.utils import RANDOM_SEED, TARGET_COLUMN
from src.preprocessing import build_preprocessing_pipeline

def run_cross_validation_audit():
    # Load processed training data
    processed_path = Path("data/processed/student_data_processed.csv")
    if not processed_path.exists():
        # Fallback to absolute path resolving relative to this file
        from src.utils import PROCESSED_DATA_DIR
        processed_path = PROCESSED_DATA_DIR / "student_data_processed.csv"
        
    df = pd.read_csv(processed_path)
    X = df.drop(columns=[TARGET_COLUMN, "student_id"])
    y = df[TARGET_COLUMN]
    
    # Split training data using the same split to align with model training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    
    preprocessor, _, _, _ = build_preprocessing_pipeline()
    
    # Best models configured with grid search parameters
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(max_depth=8, min_samples_split=2, n_estimators=100, random_state=RANDOM_SEED),
        "Support Vector Machine": SVC(C=10.0, gamma="scale", probability=True, random_state=RANDOM_SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="uniform")
    }
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    print("=========================================================================")
    print("5-Fold Stratified Cross-Validation Performance (Training Split, N=800)")
    print("=========================================================================")
    
    for name, clf in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv_strategy,
            scoring=["accuracy", "f1_macro"],
            n_jobs=-1
        )
        
        acc_mean = np.mean(cv_results["test_accuracy"])
        acc_std = np.std(cv_results["test_accuracy"])
        f1_mean = np.mean(cv_results["test_f1_macro"])
        f1_std = np.std(cv_results["test_f1_macro"])
        
        print(f"{name}:")
        print(f"  - Accuracy: {acc_mean:.4f} (+/- {acc_std:.4f})")
        print(f"  - Macro F1: {f1_mean:.4f} (+/- {f1_std:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    from pathlib import Path
    run_cross_validation_audit()
