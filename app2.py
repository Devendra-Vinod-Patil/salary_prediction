import streamlit as st
import joblib
import numpy as np
import os

# =========================
# Load Pickle Files
# =========================
base_path = os.path.dirname(__file__)
rfe = joblib.load(os.path.join(base_path, "rfe.pkl"))
model = joblib.load(os.path.join(base_path, "final_model.pkl"))

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Salary Prediction App", page_icon="💼", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Fill in your details to predict the expected salary.")

# =========================
# Input UI
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
# Backend Encoding
# =========================
def encode_inputs(city, industry, job_role, skills, experience, education_score, infra_score):
    # Example: simple encoding, expand to match your preprocessing
    input_data = []

    # Encode city
    all_cities = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune",
                  "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Indore", "Raipur", "Ranchi"]
    city_encoded = [1 if c == city else 0 for c in all_cities]
    input_data.extend(city_encoded)

    # Encode industry
    all_industries = ["Tech", "Finance", "Education", "Healthcare", "Manufacturing", "Retail", "Pharma", "Others"]
    industry_encoded = [1 if i == industry else 0 for i in all_industries]
    input_data.extend(industry_encoded)

    # Encode job role
    all_roles = ["Software Engineer", "Data Scientist", "AI/ML", "Business Development",
                 "Sales", "Operations", "Manager", "Others"]
    role_encoded = [1 if r == job_role else 0 for r in all_roles]
    input_data.extend(role_encoded)

    # Encode skills (multi-hot encoding)
    all_skills = ["Python", "Java", "C++", "SQL", "Machine Learning", "Deep Learning",
                  "TensorFlow", "NLP", "Excel", "Communication", "Project Management", "Data Visualization"]
    skills_encoded = [1 if s in skills else 0 for s in all_skills]
    input_data.extend(skills_encoded)

    # Numeric values
    input_data.append(experience)
    input_data.append(education_score)
    input_data.append(infra_score)

    return np.array(input_data).reshape(1, -1)

# =========================
# Prediction
# =========================
if st.button("Predict Salary"):
    try:
        input_array = encode_inputs(city, industry, job_role, skills, experience, education_score, infra_score)

        # RFE expects the same feature size
        input_array = rfe.transform(input_array)

        prediction = model.predict(input_array)[0]
        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
