import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
# This model was trained on one-hot encoded categorical features and scaled numerical features.
# We will replicate the feature engineering process for prediction.
try:
    model = joblib.load('final_model.pkl')
    # Extract feature names from the model, which knows the exact column order and names it was trained on.
    MODEL_FEATURES = model.feature_names_in_
except FileNotFoundError:
    st.error("Error: `final_model.pkl` not found. Please make sure the model file is in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --- Data Scaling ---
# The model was trained on data scaled with StandardScaler. Since the scaler object
# was not saved, we will manually scale the inputs using the mean and standard deviation
# from the training data analysis found in the notebook. This is crucial for accurate predictions.
SCALING_STATS = {
    'Talent_Inflow': {'mean': 1305.96, 'std': 783.72},
    'Talent_Outflow': {'mean': 1300.92, 'std': 498.17},
    'Infrastructure_Score': {'mean': 67.17, 'std': 11.32},
    'Smart_City_Investment': {'mean': 3922.67, 'std': 2180.64},
    'GCC_Presence': {'mean': 0.299, 'std': 0.458},
    'MSME_Growth_Rate': {'mean': 7.27, 'std': 3.38},
    'Unemployment_Rate': {'mean': 7.48, 'std': 2.73},
    'Education_Hubs': {'mean': 14.78, 'std': 7.29},
    'Cost_of_Living_Index': {'mean': 79.98, 'std': 11.50},
    'Job_ID': {'mean': 6000.5, 'std': 3464.24},
    # Estimated stats for date parts as they were included in the numerical features for scaling
    'day': {'mean': 15.7, 'std': 8.8},
    'month': {'mean': 6.5, 'std': 3.45},
    'year': {'mean': 2023.9, 'std': 0.8},
    'weekday': {'mean': 3.0, 'std': 2.0},
    'quarter': {'mean': 2.5, 'std': 1.12}
}


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
            .correction-note {
                font-size: 14px;
                color: #555;
                text-align: center;
                margin-top: 10px;
            }
        </style>
    """, unsafe_allow_html=True)


    st.markdown('<h1 class="title">Indian Non-Metro Job Salary Prediction</h1>', unsafe_allow_html=True)
    st.info("This app predicts the salary based on job-related features. Please fill in the details below to match the model's requirements.")

    # --- Create Input Fields for User ---
    # Based on the analysis of the provided notebook

    # Categorical Features (with options extracted from the notebook)
    cities = ['Raipur', 'Vijayawada', 'Jodhpur', 'Visakhapatnam', 'Madurai', 'Ranchi', 'Rajkot', 'Agra', 'Vadodara', 'Nashik']
    industries = ['Tech', 'Pharma', 'Others', 'Retail', 'Education', 'Manufacturing', 'Healthcare', 'Finance']
    job_roles = ['Business Development', 'Engineering', 'AI/ML', 'Operations', 'Data Science', 'Others', 'Sales', 'Retail']
    exp_levels = ['Entry-Level', 'Mid-Level', 'Senior-Level', 'Internship']
    company_names = ['Reliance Retail', 'Others', 'HCLTech']
    # The model was trained on very specific skill combinations. A dropdown is the only way to ensure valid input.
    skill_sets = [
        'Market Analysis, Sales, CRM', 'Sales, Market Analysis, CRM', 'CRM, Market Analysis, Sales',
        'CRM, Sales, Market Analysis', 'Market Analysis, CRM, Sales', 'Python, Java', 'SQL, Python',
        'SQL, Statistics', 'Python, Deep Learning, TensorFlow, NLP', 'Python, Data Visualization, Statistics',
        'Customer Service, English', 'MS Office, Customer Service', 'Negotiation, Communication',
        'SQL, Python, Data Visualization', 'Data Visualization, Statistics', 'Java, Project Management'
    ]


    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Job & Company Details")
        city = st.selectbox('City', options=cities)
        industry = st.selectbox('Industry', options=industries)
        job_role = st.selectbox('Job Role', options=job_roles)
        exp_level = st.selectbox('Experience Level', options=exp_levels)
        company_name = st.selectbox('Company Name', options=company_names)
        skill_set = st.selectbox('Skill Set', options=skill_sets)

    with col2:
        st.subheader("Economic & Infrastructure Factors")
        talent_inflow = st.number_input('Talent Inflow', min_value=0, value=1300)
        talent_outflow = st.number_input('Talent Outflow', min_value=0, value=1300)
        infra_score = st.number_input('Infrastructure Score', min_value=0.0, max_value=100.0, value=67.0, step=0.1)
        smart_city_inv = st.number_input('Smart City Investment (in Cr)', min_value=0.0, value=3900.0, step=100.0)
        msme_growth = st.number_input('MSME Growth Rate (%)', min_value=0.0, value=7.2, step=0.1)
        unemployment_rate = st.number_input('Unemployment Rate (%)', min_value=0.0, value=7.5, step=0.1)
        education_hubs = st.number_input('Number of Education Hubs', min_value=0, value=15)
        cost_of_living = st.number_input('Cost of Living Index', min_value=0.0, max_value=150.0, value=80.0, step=0.1)
        gcc_presence = st.selectbox('GCC Presence', options=[0, 1], help="0 if No, 1 if Yes")

    with col3:
        st.subheader("Other Model Inputs")
        st.warning("The model was trained with some specific features like Job ID and date components. Please provide them as well.")
        # The model file includes these features, so we must provide them.
        job_id = st.number_input('Job ID (Identifier)', min_value=1, value=6000)
        day = st.number_input('Day (1-31)', min_value=1, max_value=31, value=15)
        month = st.number_input('Month (1-12)', min_value=1, max_value=12, value=6)
        year = st.number_input('Year', min_value=2020, max_value=2030, value=2024)
        weekday = st.number_input('Weekday (0=Mon, 6=Sun)', min_value=0, max_value=6, value=3)
        quarter = st.number_input('Quarter (1-4)', min_value=1, max_value=4, value=2)


    if st.button('Predict Salary'):
        # --- Preprocess the input to match the model's training data ---

        # 1. Create a dictionary of the raw user inputs
        input_data = {
            'City': city,
            'Industry': industry,
            'Job_Role': job_role,
            'Skill_Set': skill_set,
            'Experience_Level': exp_level,
            'Company_Name': company_name,
            'weekday_name': 'Monday', # Placeholder, will be one-hot encoded
            'season': 'Summer',       # Placeholder, will be one-hot encoded
            'Job_ID': job_id,
            'Talent_Inflow': talent_inflow,
            'Talent_Outflow': talent_outflow,
            'Infrastructure_Score': infra_score,
            'Smart_City_Investment': smart_city_inv,
            'GCC_Presence': gcc_presence,
            'MSME_Growth_Rate': msme_growth,
            'Unemployment_Rate': unemployment_rate,
            'Education_Hubs': education_hubs,
            'Cost_of_Living_Index': cost_of_living,
            'day': day,
            'month': month,
            'year': year,
            'weekday': weekday,
            'quarter': quarter,
        }

        input_df_raw = pd.DataFrame([input_data])
        final_input_df = pd.DataFrame(columns=MODEL_FEATURES)
        final_input_df.loc[0] = 0.0 # Initialize with float to avoid dtype issues


        # 2. Populate the DataFrame with user input
        # One-hot encode categorical features
        for col in input_df_raw.select_dtypes(include='object').columns:
            feature_name = f"{col}_{input_df_raw[col].iloc[0]}"
            if feature_name in final_input_df.columns:
                final_input_df.loc[0, feature_name] = 1.0

        # Scale and fill numerical features
        for col in input_df_raw.select_dtypes(include=np.number).columns:
             if col in final_input_df.columns:
                # Apply standard scaling manually
                mean = SCALING_STATS.get(col, {}).get('mean', 0)
                std = SCALING_STATS.get(col, {}).get('std', 1)
                # Avoid division by zero
                if std > 0:
                    scaled_value = (input_df_raw.loc[0, col] - mean) / std
                    final_input_df.loc[0, col] = scaled_value
                else:
                    final_input_df.loc[0, col] = input_df_raw.loc[0, col] # Fallback to raw value


        final_input_df = final_input_df.astype(float)
        final_input_df = final_input_df.fillna(0.0)

        # --- Make Prediction ---
        try:
            prediction = model.predict(final_input_df)[0]
            
            correction_note = ""
            # --- Add a safety net for unrealistic predictions ---
            if prediction < 300000:
                prediction = 300000 # Minimum salary from the dataset
                correction_note = "<p class='correction-note'>Note: The model produced an unrealistic prediction for these inputs. The result has been capped at the minimum reasonable salary.</p>"

            st.markdown(f'<div class="result-box"><p class="result-text">Predicted Annual Salary: ₹ {prediction:,.2f}</p>{correction_note}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure all inputs are correct. The model is sensitive to the exact feature set it was trained on.")


if __name__ == '__main__':
    main()

