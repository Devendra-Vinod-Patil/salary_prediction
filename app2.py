import streamlit as st
import joblib
import pandas as pd
import os

# =========================
# Load Trained Model Only
# =========================
base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, "final_model.pkl"))

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Salary Prediction App", page_icon="💼", layout="centered")
st.title("💼 Salary Prediction App (No Pipeline)")
st.write("Fill in your details to predict the expected salary.")

# -------------------------
# Input Fields
# -------------------------
city = st.selectbox("City", [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Indore", "Raipur", "Ranchi"
])

industry = st.selectbox("Industry", [
    "Tech", "Finance", "Education", "Healthcare", "Manufacturing",
    "Retail", "Pharma", "Others"
])

job_role = st.selectbox("Job Role", [
    "Software Engineer", "Data Scientist", "AI/ML", "Business Development",
    "Sales", "Operations", "Manager", "Others"
])

skills = st.multiselect("Skills", [
    "Python", "Java", "C++", "SQL", "Machine Learning", "Deep Learning",
    "TensorFlow", "NLP", "Excel", "Communication", "Project Management", "Data Visualization"
])

experience = st.slider("Years of Experience", 0, 30, 1)
education_hubs = st.number_input("Education Hubs (count)", min_value=0, step=1)
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
            "Skills": ", ".join(skills),  # treat skills as one string if model trained like that
            "Experience": experience,
            "Education_Hubs": education_hubs,
            "Infrastructure_Score": infra_score
        }])

        # Direct prediction without preprocessing
        prediction = model.predict(input_data)[0]

        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
