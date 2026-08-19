import joblib
import pandas as pd
from src.utils import MODELS_DIR, CLASS_MAPPING

def test_single_prediction():
    # Load the best model pipeline
    model_path = MODELS_DIR / "best_student_model.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    print("Successfully loaded model pipeline!")
    
    # Define a test sample matching our core schema
    test_sample = pd.DataFrame({
        "age": [20],
        "gender": ["Female"],
        "parent_education": ["Bachelor"],
        "family_income": ["Medium"],
        "internet_access": ["Yes"],
        "attendance_percentage": [92.0],
        "daily_study_hours": [6.0],
        "assignment_score": [85.0],
        "previous_semester_marks": [80.0],
        "class_participation": [75.0],
        "sleep_hours": [8.0],
        "extracurricular_activities": ["Yes"]
    })
    
    # Calculate engineered features using the exact same formulas
    test_sample["study_intensity"] = (test_sample["daily_study_hours"] * test_sample["assignment_score"]) / 100.0
    test_sample["attendance_category"] = "High"  # Binned: > 90%
    test_sample["academic_consistency"] = 100.0 - abs(test_sample["previous_semester_marks"] - test_sample["assignment_score"])
    
    extracurricular_indicator = 100.0  # Yes -> 100
    test_sample["engagement_score"] = (
        0.4 * test_sample["attendance_percentage"] + 
        0.4 * test_sample["class_participation"] + 
        0.2 * extracurricular_indicator
    )
    
    # Run prediction
    pred = model.predict(test_sample)[0]
    probs = model.predict_proba(test_sample)[0]
    
    print(f"Test Student Sample:\n{test_sample.T}\n")
    print(f"Predicted Performance: {CLASS_MAPPING[pred]} (Class {pred})")
    print("Class Probabilities:")
    for cls_idx, prob in enumerate(probs):
        print(f"  - {CLASS_MAPPING[cls_idx]}: {prob * 100:.2f}%")

if __name__ == "__main__":
    test_single_prediction()
