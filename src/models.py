import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.utils import get_logger, RANDOM_SEED, TARGET_COLUMN, CLASS_ORDER, MODELS_DIR, REPORTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.features import add_engineered_features
from src.preprocessing import clean_and_validate_data, build_preprocessing_pipeline
from src.evaluation import (
    calculate_metrics, 
    save_confusion_matrix_plot, 
    save_roc_curves_plot, 
    save_feature_importance_plot
)

logger = get_logger("models")

def run_model_pipeline():
    """
    Executes the entire machine learning pipeline:
    1. Loads raw synthetic data.
    2. Cleans and performs domain validation on the dataset.
    3. Engineers new student features.
    4. Performs an 80/20 train-test stratified split.
    5. Builds the column preprocessing transformer.
    6. For each of the 5 classifiers, runs GridSearchCV with 5-fold Stratified Cross-Validation.
    7. Evaluates performance on the holdout test dataset.
    8. Identifies and saves the overall best model pipeline (preprocessor + classifier).
    9. Generates evaluation reports and plots.
    """
    raw_path = RAW_DATA_DIR / "student_data_raw.csv"
    if not raw_path.exists():
        logger.error(f"Raw data file not found at {raw_path}. Run data_generation.py first.")
        return
        
    logger.info("Step 1: Loading raw data...")
    df_raw = pd.read_csv(raw_path)
    
    logger.info("Step 2: Cleaning data and applying domain validation...")
    df_clean = clean_and_validate_data(df_raw)
    
    logger.info("Step 3: Engineering student features...")
    df_features = add_engineered_features(df_clean)
    
    # Save the processed dataset for documentation & reproducibility
    processed_path = PROCESSED_DATA_DIR / "student_data_processed.csv"
    df_features.to_csv(processed_path, index=False)
    logger.info(f"Processed dataset saved to {processed_path} (Shape: {df_features.shape})")
    
    # Separate features and target
    # Ensure final_performance is dropped from features to prevent leakage!
    X = df_features.drop(columns=[TARGET_COLUMN, "student_id"])
    y = df_features[TARGET_COLUMN]
    
    # Check if target contains missing values (it shouldn't, but let's drop them if it does)
    if y.isna().any():
        logger.warning("Target contains NaN values. Dropping rows with missing target.")
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
    # Split dataset into train and test sets (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    
    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Get preprocessing transformer
    preprocessor, num_cols, nom_cols, ord_cols = build_preprocessing_pipeline()
    
    # Define models
    models_config = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs"]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=RANDOM_SEED),
            "params": {
                "classifier__max_depth": [3, 5, 8, None],
                "classifier__min_samples_split": [2, 5, 10]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=RANDOM_SEED),
            "params": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [5, 8, None],
                "classifier__min_samples_split": [2, 5]
            }
        },
        "SVM": {
            "model": SVC(probability=True, random_state=RANDOM_SEED),
            "params": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__gamma": ["scale", "auto"]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "classifier__n_neighbors": [3, 5, 7, 11],
                "classifier__weights": ["uniform", "distance"]
            }
        }
    }
    
    results_summary = {}
    best_pipelines = {}
    
    logger.info("Step 4: Beginning model training & hyperparameter tuning...")
    
    # Set up cross-validation structure
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for name, config in models_config.items():
        logger.info(f"Training {name}...")
        
        # Create pipeline: preprocessor + classifier
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", config["model"])
        ])
        
        # Grid Search with Stratified 5-Fold Cross-Validation, optimizing for macro F1
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config["params"],
            cv=cv_strategy,
            scoring="f1_macro",
            n_jobs=-1
        )
        
        # Fit GridSearch (preprocessor fits ONLY on training data!)
        grid_search.fit(X_train, y_train)
        
        best_estimator = grid_search.best_estimator_
        best_pipelines[name] = best_estimator
        
        logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
        logger.info(f"Cross-Validation Macro F1 Score: {grid_search.best_score_:.4f}")
        
        # Evaluate on the holdout test set
        y_pred = best_estimator.predict(X_test)
        y_prob = best_estimator.predict_proba(X_test)
        
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics["best_params"] = grid_search.best_params_
        metrics["cv_macro_f1"] = float(grid_search.best_score_)
        
        results_summary[name] = metrics
        
        logger.info(f"{name} Test Set Evaluation:")
        logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  - Precision: {metrics['precision']:.4f}")
        logger.info(f"  - Recall: {metrics['recall']:.4f}")
        logger.info(f"  - Macro F1: {metrics['f1_macro']:.4f}")
        logger.info(f"  - ROC-AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "  - ROC-AUC: N/A")
        
        # Save plots for each model
        save_confusion_matrix_plot(
            y_test, y_pred, 
            model_name=name, 
            save_path=REPORTS_DIR / f"confusion_matrix_{name}.png"
        )
        save_roc_curves_plot(
            y_test, y_prob, 
            model_name=name, 
            save_path=REPORTS_DIR / f"roc_curve_{name}.png"
        )
        
    # Compare models to select the absolute best by Holdout Test F1-Macro
    best_model_name = max(results_summary, key=lambda k: results_summary[k]["f1_macro"])
    logger.info(f"\n=======================================================")
    logger.info(f"Best Model Selected: {best_model_name}")
    logger.info(f"Test Holdout F1-Macro Score: {results_summary[best_model_name]['f1_macro']:.4f}")
    logger.info(f"=======================================================")
    
    # Save the best model (which contains the full pipeline!)
    best_model_path = MODELS_DIR / "best_student_model.joblib"
    joblib.dump(best_pipelines[best_model_name], best_model_path)
    logger.info(f"Serialized best model pipeline saved to {best_model_path}")
    
    # Save the fitted preprocessing pipeline separately for potential standalone deployment
    preprocessor_path = MODELS_DIR / "data_pipeline.joblib"
    joblib.dump(best_pipelines[best_model_name].named_steps["preprocessor"], preprocessor_path)
    logger.info(f"Fitted preprocessing pipeline saved to {preprocessor_path}")
    
    # Save comparison metrics report
    comparison_path = REPORTS_DIR / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    logger.info(f"Saved model comparison report to {comparison_path}")
    
    # Generate Feature Importance plot for the best model
    # Extract names from the fitted preprocessor
    fitted_preprocessor = best_pipelines[best_model_name].named_steps["preprocessor"]
    raw_feature_names = fitted_preprocessor.get_feature_names_out()
    
    # Clean the naming prefixes (num__, nom__, ord__) for readable visualization
    cleaned_feature_names = [name.split("__")[-1] for name in raw_feature_names]
    
    # Map back categorical suffixes for clarity (e.g. internet_access_Yes)
    save_feature_importance_plot(
        best_pipelines[best_model_name],
        feature_names=cleaned_feature_names,
        model_name=best_model_name,
        save_path=REPORTS_DIR / "feature_importance_best.png"
    )
    
    logger.info("Model pipeline run complete!")

if __name__ == "__main__":
    run_model_pipeline()
