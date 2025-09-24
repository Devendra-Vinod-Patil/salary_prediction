import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Note: The original model 'final_model.pkl' is not used for prediction because the
# feature set has been changed based on user feedback. A new heuristic-based
# prediction logic is implemented below.
# model = joblib.load('final_model.pkl')
# MODEL_FEATURES = model.feature_names_in_

def main():
    """Main function to run the Streamlit application."""

    st.set_page_config(layout="wide", page_title="Salary Prediction App")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .main {
                background-color: #f5f5f5;
            }
            .stApp {
                background-color: #f5f5f5;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 12px;
                padding: 10px 24px;
                font-size: 16px;
                border: none;
                transition: all 0.3s;
            }
            .stButton>button:hover {
                background-color: #45a049;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            }
            .title {
                text-align: center;
                color: #333;
                font-family: 'Arial Black', Gadget, sans-serif;
            }
            .result-box {
                background-color: #e8f5e9;
                border-left: 6px solid #4CAF50;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
            }
            .result-text {
                font-size: 24px;
                font-weight: bold;
                color: #2e7d32;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)


    st.markdown('<h1 class="title">Indian Job Salary Prediction</h1>', unsafe_allow_html=True)
    st.info("This app predicts salary based on the recommended features. Please fill in the details below.")
    st.warning("The original machine learning model is incompatible with these new features. This app uses a rule-based calculation for demonstration purposes.", icon="⚠️")

    # --- Create Input Fields for User based on new requirements ---

    # Define options for the new categorical features
    job_roles = ['Business Development', 'Engineering', 'AI/ML', 'Operations', 'Data Science', 'Sales', 'Retail', 'Marketing', 'HR']
    industries = ['Tech', 'Pharma', 'Retail', 'Education', 'Manufacturing', 'Healthcare', 'Finance', 'Consulting']
    exp_levels = ['Internship', 'Entry-Level', 'Mid-Level', 'Senior-Level', 'Lead/Manager']
    company_levels = ['Startup (1-50 employees)', 'Mid-Size (51-500 employees)', 'Large (501-5000 employees)', 'MNC (5000+ employees)']
    job_functions = ['Technology', 'Sales & Marketing', 'Operations & Logistics', 'Finance & Admin', 'Human Resources']
    cities = ['Raipur', 'Vijayawada', 'Jodhpur', 'Visakhapatnam', 'Madurai', 'Ranchi', 'Rajkot', 'Agra', 'Vadodara', 'Nashik']
    work_modes = ['On-site', 'Hybrid', 'Remote']
    all_skills = [
        'Python', 'Java', 'SQL', 'Data Analysis', 'Machine Learning', 'Deep Learning', 'NLP', 'TensorFlow',
        'Project Management', 'Agile', 'Scrum', 'Sales', 'CRM', 'Market Analysis', 'Customer Service', 'MS Office',
        'Communication', 'Negotiation', 'Logistics', 'Operations Management'
    ]


    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🧠 Job & Company Details")
        job_role = st.selectbox('Job Role', options=job_roles)
        industry = st.selectbox('Industry', options=industries)
        exp_level = st.selectbox('Experience Level', options=exp_levels)
        company_level = st.selectbox('Company Level', options=company_levels)
        job_function = st.selectbox('Job Function', options=job_functions)
        selected_skills = st.multiselect('Select Skills (up to 5)', options=all_skills, max_selections=5)

    with col2:
        st.subheader("🌍 Location & Economic Context")
        city = st.selectbox('City', options=cities)
        gdp = st.number_input('City GDP (in crores)', min_value=10000, value=50000, step=1000)
        cost_of_living = st.number_input('Cost of Living Index', min_value=50.0, max_value=150.0, value=80.0, step=0.1)

        st.subheader("🖥️ Infrastructure & Work Mode")
        internet_penetration = st.slider('Internet Penetration (%)', min_value=0, max_value=100, value=60)
        work_mode = st.selectbox('Work Mode', options=work_modes)


    with col3:
        st.subheader("📅 Time & Calibration Inputs")
        base_pay = st.number_input('Known Base Pay (Optional, helps anchor prediction)', min_value=0, value=0)
        month = st.select_slider('Hiring Month / Quarter', options=['Jan', 'Feb', 'Mar', 'Q1', 'Apr', 'May', 'Jun', 'Q2', 'Jul', 'Aug', 'Sep', 'Q3', 'Oct', 'Nov', 'Dec', 'Q4'], value='Jun')


    if st.button('Predict Salary'):
        # --- Preprocess and Predict based on new heuristic logic ---

        # 1. Base Salary Calculation
        if base_pay > 0:
            predicted_salary = base_pay
        else:
            # Start with a base determined by industry
            industry_base = {'Tech': 600000, 'Finance': 550000, 'Healthcare': 500000, 'Manufacturing': 450000, 'Retail': 400000, 'Education': 420000, 'Pharma': 520000, 'Consulting': 650000}
            predicted_salary = industry_base.get(industry, 480000)

        # 2. Adjust for Experience Level
        exp_multiplier = {'Internship': 0.4, 'Entry-Level': 1.0, 'Mid-Level': 1.6, 'Senior-Level': 2.5, 'Lead/Manager': 3.5}
        predicted_salary *= exp_multiplier.get(exp_level, 1.0)

        # 3. Adjust for Company Level
        company_multiplier = {'Startup (1-50 employees)': 0.9, 'Mid-Size (51-500 employees)': 1.05, 'Large (501-5000 employees)': 1.2, 'MNC (5000+ employees)': 1.35}
        predicted_salary *= company_multiplier.get(company_level, 1.0)

        # 4. Add bonus for skills
        skill_bonus = len(selected_skills) * 25000 # Add 25k for each skill
        predicted_salary += skill_bonus

        # 5. Adjust for Work Mode
        work_mode_adjust = {'Remote': 1.05, 'Hybrid': 1.0, 'On-site': 0.98} # Slight premium for remote
        predicted_salary *= work_mode_adjust.get(work_mode, 1.0)

        # 6. Adjust for Cost of Living
        # Normalize CoL against a baseline of 80
        col_adjust = cost_of_living / 80.0
        predicted_salary *= col_adjust

        st.markdown(f'<div class="result-box"><p class="result-text">Predicted Annual Salary: ₹ {predicted_salary:,.2f}</p></div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()

