import streamlit as st
import joblib
import pandas as pd
import os

# =========================
# Load Models & Files
# =========================
base_path = os.path.dirname(__file__)

# Load preprocessing pipeline
pipeline = joblib.load(os.path.join(base_path, "salary_pipeline.pkl"))

# Load final trained model
final_model = joblib.load(os.path.join(base_path, "final_model.pkl"))

# Load feature names and RFE object
feature_names = joblib.load(os.path.join(base_path, "feature_names.pkl"))
rfe = joblib.load(os.path.join(base_path, "rfe.pkl"))

# Extract selected features from RFE
selected_features = [f for f, s in zip(feature_names, rfe.support_) if s]

# =========================
# Streamlit App UI
# =========================
st.set_page_config(page_title="Salary Prediction App", page_icon="💼", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Fill in your details to predict the expected salary.")

# =========================
# Input Fields
# =========================
city = st.selectbox("Select City", [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Indore", "Raipur", "Ranchi"
])

industry = st.selectbox("Industry", [
    "Tech", "Finance", "Education", "Healthcare", "Manufacturing", "Retail", "Pharma", "Others"
])

job_role = st.selectbox("Job Role", [
    "Software Engineer", "Data Scientist", "AI/ML", "Business Development",
    "Sales", "Operations", "Manager", "Others"
])

skills = st.multiselect("Skills", [
    "Python", "Java", "C++", "SQL", "Machine Learning", "Deep Learning", "TensorFlow",
    "NLP", "Excel", "Communication", "Project Management", "Data Visualization"
])

experience = st.slider("Years of Experience", 0, 20, 1)
education_score = st.number_input("Education Hubs (count)", min_value=0, step=1)
infra_score = st.number_input("Infrastructure Score", min_value=0.0, step=0.1)

# =========================
# Prediction
# =========================
if st.button("Predict Salary"):
    try:
        # Create input DataFrame
        input_data = pd.DataFrame([{
            "City": city,
            "Industry": industry,
            "Job_Role": job_role,
            "Skills": ", ".join(skills),
            "Experience": experience,
            "Education_Hubs": education_score,
            "Infrastructure_Score": infra_score
        }])

        # Keep only selected features
        input_data = input_data[[col for col in selected_features if col in input_data.columns]]

        # Apply preprocessing
        X_processed = pipeline.transform(input_data)

        # Predict with final model
        prediction = final_model.predict(X_processed)[0]

        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
