import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Paths relative to this file
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

CLASS_MAPPING = {
    0: "Poor",
    1: "Average",
    2: "Good",
    3: "Excellent"
}

CLASS_COLORS = {
    "Poor": "🔴",
    "Average": "🟡",
    "Good": "🟢",
    "Excellent": "🏆"
}

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction Dashboard")
st.markdown("""
This dashboard predicts a student's academic performance category (**Poor, Average, Good, Excellent**) 
based on scholastic, behavioral, and demographic features.
It utilizes a machine learning pipeline trained on synthetic student records and supports early academic intervention.
""")

# 1. Load the Best Model Pipeline
@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "best_student_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("⚠️ Best model not found! Please run the training pipeline first (`python src/models.py`).")
else:
    # 2. Main Interface Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📝 Input Student Metrics")
        
        # Demographic Input
        st.subheader("Demographics")
        age = st.slider("Age", 15, 25, 19)
        gender = st.selectbox("Gender", ["Male", "Female"])
        parent_education = st.selectbox(
            "Parent Education Level", 
            ["High School", "College", "Associate", "Bachelor", "Postgraduate"]
        )
        family_income = st.selectbox("Family Income Category", ["Low", "Medium", "High"])
        internet_access = st.selectbox("Internet Access at Home", ["Yes", "No"])
        
        # Behavioral Input
        st.subheader("Behavior & Scholastics")
        attendance_percentage = st.slider("Attendance Percentage (%)", 0.0, 100.0, 85.0)
        daily_study_hours = st.slider("Daily Study Hours", 0.0, 24.0, 4.5, step=0.5)
        sleep_hours = st.slider("Sleep Hours", 4.0, 16.0, 7.0, step=0.5)
        extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        
        # Academic Scores
        st.subheader("Academic History")
        assignment_score = st.slider("Current Assignment Score (0-100)", 0.0, 100.0, 75.0)
        previous_semester_marks = st.slider("Previous Semester Marks (0-100)", 0.0, 100.0, 70.0)
        class_participation = st.slider("Class Participation Score (0-100)", 0.0, 100.0, 65.0)
        
    with col2:
        st.header("🔮 Prediction Results")
        
        # Construct DataFrame from Inputs
        input_data = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "parent_education": [parent_education],
            "family_income": [family_income],
            "internet_access": [internet_access],
            "attendance_percentage": [attendance_percentage],
            "daily_study_hours": [daily_study_hours],
            "assignment_score": [assignment_score],
            "previous_semester_marks": [previous_semester_marks],
            "class_participation": [class_participation],
            "sleep_hours": [sleep_hours],
            "extracurricular_activities": [extracurricular_activities]
        })
        
        # Feature Engineering Math Logic
        # 1. Study Intensity
        study_intensity = (daily_study_hours * assignment_score) / 100.0
        
        # 2. Attendance Category
        if attendance_percentage < 75.0:
            attendance_category = "Low"
        elif attendance_percentage <= 90.0:
            attendance_category = "Moderate"
        else:
            attendance_category = "High"
            
        # 3. Academic Consistency
        academic_consistency = 100.0 - abs(previous_semester_marks - assignment_score)
        
        # 4. Engagement Score
        extracurricular_indicator = 100.0 if extracurricular_activities == "Yes" else 0.0
        engagement_score = (
            0.4 * attendance_percentage + 
            0.4 * class_participation + 
            0.2 * extracurricular_indicator
        )
        
        # Append engineered features to input dataframe
        input_data["study_intensity"] = [study_intensity]
        input_data["attendance_category"] = [attendance_category]
        input_data["academic_consistency"] = [academic_consistency]
        input_data["engagement_score"] = [engagement_score]
        
        # Make predictions
        # Note: model contains the complete scikit-learn pipeline
        prediction = model.predict(input_data)[0]
        prediction_probs = model.predict_proba(input_data)[0]
        
        predicted_class_name = CLASS_MAPPING[prediction]
        icon = CLASS_COLORS[predicted_class_name]
        
        # Display large prediction callout
        st.success(f"### Predicted Performance: {icon} **{predicted_class_name}**")
        
        # Display Probability distribution
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "Performance Category": ["Poor", "Average", "Good", "Excellent"],
            "Probability (%)": [prob * 100 for prob in prediction_probs]
        })
        st.bar_chart(prob_df, x="Performance Category", y="Probability (%)")
        
        # Display Engineered Features breakdown
        st.subheader("🛠️ Engineered Features Breakdown")
        
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.metric(
                label="Study Intensity", 
                value=f"{study_intensity:.2f}",
                help="Formulated as: (Study Hours * Assignment Score) / 100"
            )
        with ec2:
            st.metric(
                label="Attendance Category", 
                value=attendance_category,
                help="Binned as: Low (<75%), Moderate (75-90%), High (>90%)"
            )
        with ec3:
            st.metric(
                label="Academic Consistency", 
                value=f"{academic_consistency:.1f}",
                help="Formulated as: 100 - |Prev Marks - Assignment Score|"
            )
        with ec4:
            st.metric(
                label="Engagement Score", 
                value=f"{engagement_score:.1f}",
                help="Formulated as: 0.4 * Attendance + 0.4 * Participation + 0.2 * Extracurricular"
            )
            
        # Model Artifacts Section
        st.subheader("📊 Trained Model Insights")
        
        # Read model comparison report if it exists
        comparison_path = REPORTS_DIR / "model_comparison.json"
        if comparison_path.exists():
            with open(comparison_path, "r") as f:
                comp_data = json.load(f)
            
            comp_df = pd.DataFrame(comp_data).T.drop(columns=["best_params", "cv_macro_f1"], errors="ignore")
            comp_df.columns = ["Holdout Accuracy", "Holdout Precision", "Holdout Recall", "Holdout F1-Macro", "Holdout ROC-AUC"]
            st.markdown("**Comparison of Model Metrics on Holdout Test Set:**")
            st.dataframe(comp_df.style.highlight_max(axis=0, color="#d4edda"))
            
        # Show Feature Importance Chart
        feat_imp_path = REPORTS_DIR / "feature_importance_best.png"
        if feat_imp_path.exists():
            st.markdown(f"**Feature Importance Plot for Best Model:**")
            st.image(str(feat_imp_path), use_container_width=True)
            
        # Show Confusion Matrix Chart for Best Model
        best_model_name = "RandomForest" # Default fallback, overwritten by loaded json if possible
        if comparison_path.exists():
            best_model_name = max(comp_data, key=lambda k: comp_data[k]["f1_macro"])
            
        cm_path = REPORTS_DIR / f"confusion_matrix_{best_model_name}.png"
        if cm_path.exists():
            st.markdown(f"**Confusion Matrix for {best_model_name} (Holdout Set):**")
            st.image(str(cm_path), use_container_width=True)
