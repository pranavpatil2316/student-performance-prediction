import numpy as np
import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to the student performance dataset.
    
    1. Study Intensity:
       Study Intensity = (Daily Study Hours * Assignment Score) / 100.0
       Rationale: Captures the joint effect of study duration and assignment quality.
       
    2. Attendance Category:
       Attendance Category = 
         "Low"      if Attendance < 75%
         "Moderate" if 75% <= Attendance <= 90%
         "High"     if Attendance > 90%
       Rationale: Captures scholastic compliance threshold categories.
       
    3. Academic Consistency:
       Academic Consistency = 100.0 - |Previous Semester Marks - Assignment Score|
       Rationale: Measures how stable a student's marks are. Higher values mean higher consistency.
       
    4. Engagement Score:
       Engagement Score = 0.4 * Attendance Percentage + 0.4 * Class Participation + 0.2 * Extracurricular Indicator
       Where Extracurricular Indicator = 100 for "Yes" and 0 for "No".
       Rationale: Balanced measure of behavioral and interactive student involvement (bounded 0-100).
    """
    df_feat = df.copy()
    
    # 1. Study Intensity
    df_feat["study_intensity"] = (df_feat["daily_study_hours"] * df_feat["assignment_score"]) / 100.0
    
    # 2. Attendance Category
    # We assign string categories which will be encoded during preprocessing
    conditions = [
        df_feat["attendance_percentage"] < 75.0,
        (df_feat["attendance_percentage"] >= 75.0) & (df_feat["attendance_percentage"] <= 90.0),
        df_feat["attendance_percentage"] > 90.0
    ]
    choices = ["Low", "Moderate", "High"]
    # If attendance is NaN, the np.select will assign default ('Unknown')
    df_feat["attendance_category"] = np.select(conditions, choices, default="Unknown")
    # Ensure NaN values are preserved as object/None for the Imputer to handle
    df_feat.loc[df_feat["attendance_percentage"].isna(), "attendance_category"] = np.nan
    
    # 3. Academic Consistency
    df_feat["academic_consistency"] = 100.0 - (df_feat["previous_semester_marks"] - df_feat["assignment_score"]).abs()
    
    # 4. Engagement Score
    extracurricular_indicator = np.where(df_feat["extracurricular_activities"] == "Yes", 100.0, 0.0)
    # Handle possible NaN in extracurricular activities if any
    extracurricular_indicator = np.where(df_feat["extracurricular_activities"].isna(), np.nan, extracurricular_indicator)
    
    df_feat["engagement_score"] = (
        0.4 * df_feat["attendance_percentage"] + 
        0.4 * df_feat["class_participation"] + 
        0.2 * extracurricular_indicator
    )
    
    return df_feat
