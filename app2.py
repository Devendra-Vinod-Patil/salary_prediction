import streamlit as st
import joblib
import numpy as np
import os

# =========================
# Load Pickle Files
# =========================
base_path = os.path.dirname(__file__)

feature_names = joblib.load(os.path.join(base_path, "feature_names.pkl"))
rfe = joblib.load(os.path.join(base_path, "rfe.pkl"))
model = joblib.load(os.path.join(base_path, "final_model.pkl"))

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Salary Prediction App", page_icon="💼", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Fill in your details below to predict the expected salary.")

# =========================
# Feature Inputs
# =========================

user_input = {}

# Example mappings (adjust based on your dataset)
categorical_options = {
    "state": ["Maharashtra", "Gujarat", "Karnataka", "Delhi", "Tamil Nadu"],
    "company_role": ["Software Engineer", "Data Analyst", "Manager", "Developer", "Intern"],
    "skills": [
        "Python", "Java", "C++", "C#", "JavaScript", "SQL", "R", "PHP",
        "HTML/CSS", "Excel", "Communication", "Machine Learning", "Deep Learning",
        "Data Science", "Cloud Computing", "AWS", "Azure", "Google Cloud",
        "Tableau", "Power BI", "Git/GitHub", "Linux", "Networking"
    ]
}


for feature in feature_names:
    if feature in categorical_options:
        user_input[feature] = st.selectbox(f"{feature}", categorical_options[feature])
    else:
        user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=1.0)

# =========================
# Prediction
# =========================

if st.button("Predict Salary"):
    try:
        # Convert to array
        input_values = [user_input[feat] for feat in feature_names]

        # Encode categorical values to numeric (basic handling)
        input_encoded = []
        for feat, val in zip(feature_names, input_values):
            if feat in categorical_options:
                input_encoded.append(categorical_options[feat].index(val))  # simple label encoding
            else:
                input_encoded.append(val)

        input_array = np.array(input_encoded).reshape(1, -1)

        # Apply RFE transform
        input_array = rfe.transform(input_array)

        # Predict
        prediction = model.predict(input_array)[0]
        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
