import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("final_model.pkl")
MODEL_FEATURES = model.feature_names_in_

def main():
    st.set_page_config(page_title="Salary Prediction", layout="wide")
    st.title("💼 Indian Non-Metro Job Salary Prediction")

    st.info("Select job details below. Dropdowns are used for categorical fields.")

    # --- Dropdowns for categorical variables ---
    cities = ["Madurai", "Nashik", "Raipur", "Rajkot", "Ranchi", 
              "Vadodara", "Vijayawada", "Visakhapatnam"]
    industries = ["Education", "Finance", "Healthcare", "Manufacturing", "Pharma", "Retail", "Tech", "Others"]
    job_roles = ["AI/ML", "Business Development", "Data Science", "Engineering", "Operations", "Retail", "Sales", "Others"]
    skills = [
        "Python", "SQL", "Java", "Machine Learning", "Deep Learning", "NLP", 
        "TensorFlow", "Project Management", "Sales", "CRM", "Customer Service", 
        "MS Office", "Communication", "Negotiation", "Logistics", "Operations Management"
    ]
    exp_levels = ["Internship", "Entry-Level", "Mid-Level", "Senior-Level"]
    companies = ["HCLTech", "Reliance Retail", "Others"]

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", cities)
        industry = st.selectbox("Industry", industries)
        job_role = st.selectbox("Job Role", job_roles)
        exp_level = st.selectbox("Experience Level", exp_levels)
        company = st.selectbox("Company", companies)

    with col2:
        selected_skills = st.multiselect("Skills (choose up to 5)", skills, max_selections=5)
        cost_index = st.number_input("Cost of Living Index", min_value=50.0, max_value=150.0, value=80.0, step=0.1)
        unemployment_rate = st.slider("Unemployment Rate (%)", min_value=0.0, max_value=20.0, value=5.0)
        infra_score = st.slider("Infrastructure Score", min_value=0.0, max_value=100.0, value=50.0)

    if st.button("Predict Salary"):
        # --- Convert selections to model features (one-hot encoding) ---
        input_dict = {col: 0 for col in MODEL_FEATURES}

        # Encode categorical features
        if f"City_{city}" in input_dict: input_dict[f"City_{city}"] = 1
        if f"Industry_{industry}" in input_dict: input_dict[f"Industry_{industry}"] = 1
        if f"Job_Role_{job_role}" in input_dict: input_dict[f"Job_Role_{job_role}"] = 1
        if f"Experience_Level_{exp_level}" in input_dict: input_dict[f"Experience_Level_{exp_level}"] = 1
        if f"Company_Name_{company}" in input_dict: input_dict[f"Company_Name_{company}"] = 1
        for skill in selected_skills:
            col_name = f"Skill_Set_{skill}"
            if col_name in input_dict:
                input_dict[col_name] = 1

        # Numeric features
        if "Cost_of_Living_Index" in input_dict: input_dict["Cost_of_Living_Index"] = cost_index
        if "Unemployment_Rate" in input_dict: input_dict["Unemployment_Rate"] = unemployment_rate
        if "Infrastructure_Score" in input_dict: input_dict["Infrastructure_Score"] = infra_score

        # Convert to DataFrame
        X_input = pd.DataFrame([input_dict])

        # Predict
        salary = model.predict(X_input)[0]

        st.success(f"Predicted Annual Salary: ₹ {salary:,.2f}")

if __name__ == "__main__":
    main()
