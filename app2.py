import streamlit as st
import pandas as pd
import joblib

# ====================
# Load trained pipeline/model
# ====================
model = joblib.load("final_model.pkl")
MODEL_FEATURES = model.feature_names_in_

def main():
    st.set_page_config(page_title="💼 Salary Prediction App", layout="wide")
    st.title("💼 Salary Prediction App")
    st.info("Predict the correct salary for Indian non-metro jobs using a trained ML model.")

    # --- Dropdowns for categorical inputs ---
    cities = ["Madurai", "Nashik", "Raipur", "Rajkot", "Ranchi", "Vadodara", "Vijayawada", "Visakhapatnam"]
    industries = ["Education", "Finance", "Healthcare", "Manufacturing", "Pharma", "Retail", "Tech", "Others"]
    job_roles = ["AI/ML", "Business Development", "Data Science", "Engineering", "Operations", "Retail", "Sales", "Others"]
    exp_levels = ["Internship", "Entry-Level", "Mid-Level", "Senior-Level"]
    companies = ["HCLTech", "Reliance Retail", "Others"]

    # Skill set options must match training columns (combined features)
    skill_sets = [f for f in MODEL_FEATURES if f.startswith("Skill_Set_")]
    
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", cities)
        industry = st.selectbox("Industry", industries)
        job_role = st.selectbox("Job Role", job_roles)
        exp_level = st.selectbox("Experience Level", exp_levels)
        company = st.selectbox("Company", companies)

    with col2:
        skill_combo = st.selectbox("Skill Set", skill_sets)
        cost_index = st.number_input("Cost of Living Index", min_value=50.0, max_value=150.0, value=80.0, step=0.1)
        unemployment_rate = st.slider("Unemployment Rate (%)", min_value=0.0, max_value=20.0, value=5.0)
        infra_score = st.slider("Infrastructure Score", min_value=0.0, max_value=100.0, value=50.0)

    if st.button("Predict Salary"):
        # Build input dictionary with all features = 0
        input_dict = {col: 0 for col in MODEL_FEATURES}

        # Encode categorical selections
        if f"City_{city}" in input_dict: input_dict[f"City_{city}"] = 1
        if f"Industry_{industry}" in input_dict: input_dict[f"Industry_{industry}"] = 1
        if f"Job_Role_{job_role}" in input_dict: input_dict[f"Job_Role_{job_role}"] = 1
        if f"Experience_Level_{exp_level}" in input_dict: input_dict[f"Experience_Level_{exp_level}"] = 1
        if f"Company_Name_{company}" in input_dict: input_dict[f"Company_Name_{company}"] = 1
        if skill_combo in input_dict: input_dict[skill_combo] = 1

        # Numeric features
        if "Cost_of_Living_Index" in input_dict: input_dict["Cost_of_Living_Index"] = cost_index
        if "Unemployment_Rate" in input_dict: input_dict["Unemployment_Rate"] = unemployment_rate
        if "Infrastructure_Score" in input_dict: input_dict["Infrastructure_Score"] = infra_score

        # Create DataFrame with correct feature order
        X_input = pd.DataFrame([[input_dict[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)

        # Predict
        salary = model.predict(X_input)[0]

        st.success(f"Predicted Annual Salary: ₹ {salary:,.2f}")

if __name__ == "__main__":
    main()
