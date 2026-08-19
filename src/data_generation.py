import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import get_logger, RANDOM_SEED, RAW_DATA_DIR

logger = get_logger("data_generation")

def generate_synthetic_data(output_path: Path, num_records: int = 1000):
    """
    Generates a reproducible synthetic student performance dataset.
    
    1. Generates 1,000 clean base records with realistic academic and demographic distributions.
    2. Calculates a multi-class target variable ('final_performance') based on a latent score.
    3. Intentionally injects data anomalies (duplicates, missing values, domain-violating values)
       to test the cleaning and preprocessing pipelines.
    """
    np.random.seed(RANDOM_SEED)
    
    logger.info(f"Generating {num_records} base student records...")
    
    # 1. Base Demographic Features
    student_ids = np.arange(1001, 1001 + num_records)
    ages = np.random.normal(loc=19.5, scale=1.5, size=num_records).round().astype(int)
    ages = np.clip(ages, 16, 25)
    
    genders = np.random.choice(["Male", "Female"], size=num_records, p=[0.48, 0.52])
    
    parent_education_categories = ["High School", "College", "Associate", "Bachelor", "Postgraduate"]
    parent_education = np.random.choice(
        parent_education_categories, 
        size=num_records, 
        p=[0.20, 0.25, 0.15, 0.30, 0.10]
    )
    
    family_income_categories = ["Low", "Medium", "High"]
    family_income = np.random.choice(
        family_income_categories, 
        size=num_records, 
        p=[0.30, 0.55, 0.15]
    )
    
    internet_access = np.random.choice(["Yes", "No"], size=num_records, p=[0.85, 0.15])
    extracurricular_activities = np.random.choice(["Yes", "No"], size=num_records, p=[0.40, 0.60])
    
    # 2. Base Behavioral & Scholastic Features (correlated with target & each other)
    # Daily Study Hours (Gamma distribution, skewed right, average around 4.5 hours)
    study_hours = np.random.gamma(shape=4.5, scale=1.0, size=num_records)
    study_hours = np.clip(study_hours, 0.5, 12.0)
    
    # Attendance Percentage (Beta distribution, skewed left, average around 86%)
    attendance = np.random.beta(a=8, b=1.5, size=num_records) * 100
    attendance = np.clip(attendance, 35.0, 100.0)
    
    # Previous Semester Marks (correlated with study hours)
    prev_marks_base = 40 + (study_hours * 4.5) + np.random.normal(0, 10, num_records)
    prev_semester_marks = np.clip(prev_marks_base, 30.0, 100.0)
    
    # Assignment Score (correlated with study hours and attendance)
    assignment_base = 35 + (study_hours * 3.5) + (attendance * 0.2) + np.random.normal(0, 8, num_records)
    assignment_score = np.clip(assignment_base, 30.0, 100.0)
    
    # Class Participation (correlated with attendance)
    participation_base = (attendance * 0.8) + np.random.normal(0, 10, num_records)
    class_participation = np.clip(participation_base, 20.0, 100.0)
    
    # Sleep Hours (Normal distribution, mean 7, range 4 to 10)
    sleep_hours = np.random.normal(loc=7.2, scale=1.1, size=num_records)
    sleep_hours = np.clip(sleep_hours, 4.0, 10.0)
    
    # Build dataframe for clean data first to compute performance target
    df_clean = pd.DataFrame({
        "student_id": student_ids,
        "age": ages,
        "gender": genders,
        "parent_education": parent_education,
        "family_income": family_income,
        "internet_access": internet_access,
        "attendance_percentage": attendance,
        "daily_study_hours": study_hours,
        "assignment_score": assignment_score,
        "previous_semester_marks": prev_semester_marks,
        "class_participation": class_participation,
        "sleep_hours": sleep_hours,
        "extracurricular_activities": extracurricular_activities
    })
    
    # 3. Calculate Latent Score and Target variable ('final_performance')
    # Normalizing features to a common 0-100 range for the latent utility score
    att_n = df_clean["attendance_percentage"]
    study_n = np.clip(df_clean["daily_study_hours"] * 10, 0, 100)
    assign_n = df_clean["assignment_score"]
    prev_n = df_clean["previous_semester_marks"]
    part_n = df_clean["class_participation"]
    
    # Sleep utility peaks around 7.5 hours, rescaled to 0-100
    sleep_n = np.clip((df_clean["sleep_hours"] - 4.0) / 6.0 * 100.0, 0, 100)
    
    internet_n = np.where(df_clean["internet_access"] == "Yes", 100.0, 0.0)
    
    income_map = {"Low": 0.0, "Medium": 50.0, "High": 100.0}
    income_n = df_clean["family_income"].map(income_map)
    
    edu_map = {"High School": 0.0, "College": 33.3, "Associate": 50.0, "Bachelor": 75.0, "Postgraduate": 100.0}
    edu_n = df_clean["parent_education"].map(edu_map)
    
    # Calculate Latent Score (weights sum to exactly 1.00)
    latent_score = (
        0.25 * att_n + 
        0.20 * study_n + 
        0.20 * assign_n + 
        0.15 * prev_n + 
        0.08 * part_n + 
        0.05 * sleep_n + 
        0.02 * internet_n + 
        0.02 * income_n + 
        0.03 * edu_n +
        np.random.normal(loc=0, scale=7.0, size=num_records) # Controlled noise
    )
    
    # Multi-class target variable mapping: Poor=0, Average=1, Good=2, Excellent=3
    final_performance = np.zeros(num_records, dtype=int)
    final_performance[latent_score < 50.0] = 0          # Poor
    final_performance[(latent_score >= 50.0) & (latent_score < 65.0)] = 1  # Average
    final_performance[(latent_score >= 65.0) & (latent_score < 80.0)] = 2  # Good
    final_performance[latent_score >= 80.0] = 3          # Excellent
    
    df_clean["final_performance"] = final_performance
    
    # Log target distribution
    distribution = df_clean["final_performance"].value_counts().sort_index()
    logger.info(f"Target variable distributions: \n{distribution}")
    
    # 4. Inject controlled anomalies to test data cleaning pipelines
    df_dirty = df_clean.copy()
    
    # Inject Duplicates (2% = 20 records duplicated)
    dup_indices = np.random.choice(num_records, size=20, replace=False)
    duplicates = df_dirty.iloc[dup_indices].copy()
    # Add minor noise to duplicates to simulate clerical variations
    duplicates["age"] = np.clip(duplicates["age"] + np.random.choice([-1, 0, 1], size=20), 16, 25)
    df_dirty = pd.concat([df_dirty, duplicates], ignore_index=True)
    
    # Inject Missing Values (3% random NaNs in selected columns)
    cols_to_nan = ["attendance_percentage", "daily_study_hours", "assignment_score", "parent_education"]
    for col in cols_to_nan:
        nan_indices = np.random.choice(df_dirty.index, size=int(0.03 * len(df_dirty)), replace=False)
        df_dirty.loc[nan_indices, col] = np.nan
        
    # Inject Invalid Domain Values (to test domain validation before stats outlier processing)
    # A few invalid attendance percentages (e.g. -5.0, 115.0)
    attendance_invalid_idx = np.random.choice(df_dirty.index, size=5, replace=False)
    df_dirty.loc[attendance_invalid_idx[:3], "attendance_percentage"] = -10.0
    df_dirty.loc[attendance_invalid_idx[3:], "attendance_percentage"] = 120.0
    
    # A few invalid study hours (e.g. -2.0, 26.0)
    study_invalid_idx = np.random.choice(df_dirty.index, size=5, replace=False)
    df_dirty.loc[study_invalid_idx[:3], "daily_study_hours"] = -2.5
    df_dirty.loc[study_invalid_idx[3:], "daily_study_hours"] = 28.0
    
    # A few invalid sleep hours (valid domain is [4.0, 16.0], inject values like 1.5, 24.5)
    sleep_invalid_idx = np.random.choice(df_dirty.index, size=5, replace=False)
    df_dirty.loc[sleep_invalid_idx[:3], "sleep_hours"] = 1.5
    df_dirty.loc[sleep_invalid_idx[3:], "sleep_hours"] = 25.0
    
    # A few invalid assignment scores (e.g. -15.0, 130.0)
    assignment_invalid_idx = np.random.choice(df_dirty.index, size=5, replace=False)
    df_dirty.loc[assignment_invalid_idx[:3], "assignment_score"] = -15.0
    df_dirty.loc[assignment_invalid_idx[3:], "assignment_score"] = 135.0
    
    # A few invalid previous semester marks (e.g. -5.0, 110.0)
    prev_marks_invalid_idx = np.random.choice(df_dirty.index, size=5, replace=False)
    df_dirty.loc[prev_marks_invalid_idx[:3], "previous_semester_marks"] = -5.0
    df_dirty.loc[prev_marks_invalid_idx[3:], "previous_semester_marks"] = 110.0
    
    # A few invalid class participations (e.g. -8.0, 140.0)
    participation_invalid_idx = np.random.choice(df_dirty.index, size=5, replace=False)
    df_dirty.loc[participation_invalid_idx[:3], "class_participation"] = -8.0
    df_dirty.loc[participation_invalid_idx[3:], "class_participation"] = 140.0
    
    # Save raw dataset
    df_dirty.to_csv(output_path, index=False)
    logger.info(f"Raw dataset with anomalies successfully saved to {output_path} (Shape: {df_dirty.shape})")
    
if __name__ == "__main__":
    raw_data_file = RAW_DATA_DIR / "student_data_raw.csv"
    generate_synthetic_data(raw_data_file)
