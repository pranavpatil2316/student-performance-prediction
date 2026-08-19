import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib
from src.utils import get_logger, PROCESSED_DATA_DIR, MODELS_DIR

logger = get_logger("preprocessing")

class OutlierWinsorizer(BaseEstimator, TransformerMixin):
    """
    Fits outlier thresholds (quantiles) on the training set and caps values on train/test sets.
    Prevents data leakage by saving training quantiles.
    """
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.lower_quantile).to_dict()
        self.upper_bounds_ = X_df.quantile(self.upper_quantile).to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col in self.lower_bounds_ and col in self.upper_bounds_:
                X_df[col] = np.clip(X_df[col], self.lower_bounds_[col], self.upper_bounds_[col])
        return X_df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs duplicate removal and domain validation before feature engineering.
    """
    df_clean = df.copy()
    
    # 1. Duplicate Removal
    # Identify duplicate Student IDs and keep the first occurrence
    initial_shape = df_clean.shape
    df_clean = df_clean.drop_duplicates(subset=["student_id"], keep="first")
    duplicates_removed = initial_shape[0] - df_clean.shape[0]
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate student records.")

    # 2. Domain Validation
    # Convert invalid domain values to NaN so the pipeline can impute them
    domain_rules = {
        "attendance_percentage": (0.0, 100.0),
        "assignment_score": (0.0, 100.0),
        "previous_semester_marks": (0.0, 100.0),
        "class_participation": (0.0, 100.0),
        "daily_study_hours": (0.0, 24.0),
        "sleep_hours": (4.0, 16.0)  # Documented reasonable range
    }
    
    for col, (min_val, max_val) in domain_rules.items():
        invalid_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
        num_invalid = invalid_mask.sum()
        if num_invalid > 0:
            logger.info(f"Column '{col}': replacing {num_invalid} out-of-domain values with NaN.")
            df_clean.loc[invalid_mask, col] = np.nan
            
    return df_clean


def build_preprocessing_pipeline():
    """
    Builds the Scikit-learn ColumnTransformer for preprocessing.
    """
    # Feature columns grouped by type
    numerical_cols = [
        "age",
        "attendance_percentage",
        "daily_study_hours",
        "assignment_score",
        "previous_semester_marks",
        "class_participation",
        "sleep_hours",
        "study_intensity",
        "academic_consistency",
        "engagement_score"
    ]
    
    nominal_cols = ["gender", "internet_access", "extracurricular_activities"]
    
    ordinal_cols = ["family_income", "parent_education", "attendance_category"]
    
    # Categories order list for OrdinalEncoder
    ordinal_categories = [
        ["Low", "Medium", "High"],  # family_income
        ["High School", "College", "Associate", "Bachelor", "Postgraduate"],  # parent_education
        ["Low", "Moderate", "High"]  # attendance_category
    ]
    
    # 1. Numerical Pipeline (Impute -> Winsorize -> Scale)
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorizer", OutlierWinsorizer(lower_quantile=0.01, upper_quantile=0.99)),
        ("scaler", StandardScaler())
    ])
    
    # 2. Nominal Pipeline (Impute -> One-Hot Encode)
    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])
    
    # 3. Ordinal Pipeline (Impute -> Ordinal Encode -> Scale)
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=ordinal_categories)),
        ("scaler", StandardScaler())  # Standardize ordinals as they will feed into linear/distance models
    ])
    
    # Combine everything
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("nom", nominal_pipeline, nominal_cols),
            ("ord", ordinal_pipeline, ordinal_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor, numerical_cols, nominal_cols, ordinal_cols
