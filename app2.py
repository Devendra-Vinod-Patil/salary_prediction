import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_resource
def load_model_assets():
    """Loads the ML model and scaler, caching them for performance."""
    try:
        # Get the absolute path to the directory where the script is running
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(base_dir, 'final_model.pkl')
        scaler_path = os.path.join(base_dir, 'scaler.pkl')

        # Check if files exist before loading
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None # Return None to be handled later
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        # Catch any other loading errors
        st.error(f"An error occurred while loading model assets: {e}")
        return None, None

def preprocess_input(data, scaler, model_columns):
    """
    Preprocesses user input dictionary to create a DataFrame that matches the model's training format.
    """
    # Create a single-row DataFrame from the user's input data
    df = pd.DataFrame([data])

    # --- 1. Feature Engineering for Date Column ---
    df['Job_Posting_Date'] = pd.to_datetime(df['Job_Posting_Date'])
    df['day'] = df['Job_Posting_Date'].dt.day
    df['month'] = df['Job_Posting_Date'].dt.month
    df['year'] = df['Job_Posting_Date'].dt.year
    df['weekday'] = df['Job_Posting_Date'].dt.dayofweek
    df['quarter'] = df['Job_Posting_Date'].dt.quarter

    # --- 2. Scaling Numerical Features ---
    # Define the numerical columns that the scaler expects
    numerical_cols_to_scale = [
        'Job_ID', 'Talent_Inflow', 'Talent_Outflow', 'Infrastructure_Score',
        'Smart_City_Investment', 'GCC_Presence', 'MSME_Growth_Rate',
        'Unemployment_Rate', 'Education_Hubs', 'Cost_of_Living_Index', 'day',
        'month', 'year', 'weekday', 'quarter'
    ]
    # Apply the pre-fitted scaler
    df[numerical_cols_to_scale] = scaler.transform(df[numerical_cols_to_scale])

    # --- 3. One-Hot Encoding for Categorical Features ---
    # Create an empty DataFrame with all the columns the model was trained on, filled with zeros
    final_df = pd.DataFrame(columns=model_columns, index=df.index, data=0)

    # Populate the scaled numerical values into our final DataFrame
    for col in numerical_cols_to_scale:
        if col in final_df.columns:
            final_df[col] = df[col]

    # Dynamically create the one-hot encoded column names and set them to 1
    categorical_inputs = {
        'City': data['City'],
        'Industry': data['Industry'],
        'Job_Role': data['Job_Role'],
        'Experience_Level': data['Experience_Level'],
        'Skill_Set': data['Skill_Set'] 
    }
    
    for feature, value in categorical_inputs.items():
        # Construct the column name like 'City_Ranchi'
        one_hot_col = f"{feature}_{value}"
        if one_hot_col in final_df.columns:
            final_df[one_hot_col] = 1
            
    # Ensure the final column order matches the model's training data exactly
    return final_df[model_columns]


# --- Load Model and Scaler ---
model, scaler = load_model_assets()

# This list must exactly match the columns and their order from the training notebook.
# Any discrepancy will cause the prediction to fail.
MODEL_COLUMNS = ['City_Agra', 'City_Jodhpur', 'City_Madurai', 'City_Nashik', 'City_Raipur', 'City_Rajkot', 'City_Ranchi', 'City_Vadodara', 'City_Vijayawada', 'City_Visakhapatnam', 'Industry_Education', 'Industry_Finance', 'Industry_Healthcare', 'Industry_Manufacturing', 'Industry_Others', 'Industry_Pharma', 'Industry_Retail', 'Industry_Tech', 'Job_Role_AI/ML', 'Job_Role_Business Development', 'Job_Role_Data Science', 'Job_Role_Engineering', 'Job_Role_Operations', 'Job_Role_Others', 'Job_Role_Retail', 'Job_Role_Sales', 'Skill_Set_CRM, Market Analysis', 'Skill_Set_CRM, Market Analysis, Sales', 'Skill_Set_CRM, Sales', 'Skill_Set_CRM, Sales, Market Analysis', 'Skill_Set_Communication, Negotiation', 'Skill_Set_Communication, Negotiation, Sales', 'Skill_Set_Communication, Sales', 'Skill_Set_Communication, Sales, Negotiation', 'Skill_Set_Customer Service, English', 'Skill_Set_Customer Service, English, MS Office', 'Skill_Set_Customer Service, MS Office', 'Skill_Set_Customer Service, MS Office, English', 'Skill_Set_Data Visualization, Python', 'Skill_Set_Data Visualization, Python, SQL', 'Skill_Set_Data Visualization, Python, SQL, Statistics', 'Skill_Set_Data Visualization, Python, Statistics', 'Skill_Set_Data Visualization, Python, Statistics, SQL', 'Skill_Set_Data Visualization, SQL', 'Skill_Set_Data Visualization, SQL, Python', 'Skill_Set_Data Visualization, SQL, Python, Statistics', 'Skill_Set_Data Visualization, SQL, Statistics', 'Skill_Set_Data Visualization, SQL, Statistics, Python', 'Skill_Set_Data Visualization, Statistics', 'Skill_Set_Data Visualization, Statistics, Python', 'Skill_Set_Data Visualization, Statistics, Python, SQL', 'Skill_Set_Data Visualization, Statistics, SQL', 'Skill_Set_Data Visualization, Statistics, SQL, Python', 'Skill_Set_Deep Learning, NLP', 'Skill_Set_Deep Learning, NLP, Python', 'Skill_Set_Deep Learning, NLP, Python, TensorFlow', 'Skill_Set_Deep Learning, NLP, TensorFlow', 'Skill_Set_Deep Learning, NLP, TensorFlow, Python', 'Skill_Set_Deep Learning, Python', 'Skill_Set_Deep Learning, Python, NLP', 'Skill_Set_Deep Learning, Python, NLP, TensorFlow', 'Skill_Set_Deep Learning, Python, TensorFlow', 'Skill_Set_Deep Learning, Python, TensorFlow, NLP', 'Skill_Set_Deep Learning, TensorFlow', 'Skill_Set_Deep Learning, TensorFlow, NLP', 'Skill_Set_Deep Learning, TensorFlow, NLP, Python', 'Skill_Set_Deep Learning, TensorFlow, Python', 'Skill_Set_Deep Learning, TensorFlow, Python, NLP', 'Skill_Set_English, Customer Service', 'Skill_Set_English, Customer Service, MS Office', 'Skill_Set_English, MS Office', 'Skill_Set_English, MS Office, Customer Service', 'Skill_Set_Excel, Logistics', 'Skill_Set_Excel, Logistics, Operations Management', 'Skill_Set_Excel, Operations Management', 'Skill_Set_Excel, Operations Management, Logistics', 'Skill_Set_Java, Project Management', 'Skill_Set_Java, Project Management, Python', 'Skill_Set_Java, Project Management, Python, SQL', 'Skill_Set_Java, Project Management, SQL', 'Skill_Set_Java, Project Management, SQL, Python', 'Skill_Set_Java, Python', 'Skill_Set_Java, Python, Project Management', 'Skill_Set_Java, Python, Project Management, SQL', 'Skill_Set_Java, Python, SQL', 'Skill_Set_Java, Python, SQL, Project Management', 'Skill_Set_Java, SQL', 'Skill_Set_Java, SQL, Project Management', 'Skill_Set_Java, SQL, Project Management, Python', 'Skill_Set_Java, SQL, Python', 'Skill_Set_Java, SQL, Python, Project Management', 'Skill_Set_Logistics, Excel', 'Skill_Set_Logistics, Excel, Operations Management', 'Skill_Set_Logistics, Operations Management', 'Skill_Set_Logistics, Operations Management, Excel', 'Skill_Set_MS Office, Customer Service', 'Skill_Set_MS Office, Customer Service, English', 'Skill_Set_MS Office, English', 'Skill_Set_MS Office, English, Customer Service', 'Skill_Set_Market Analysis, CRM', 'Skill_Set_Market Analysis, CRM, Sales', 'Skill_Set_Market Analysis, Sales', 'Skill_Set_Market Analysis, Sales, CRM', 'Skill_Set_NLP, Deep Learning', 'Skill_Set_NLP, Deep Learning, Python', 'Skill_Set_NLP, Deep Learning, Python, TensorFlow', 'Skill_Set_NLP, Deep Learning, TensorFlow', 'Skill_Set_NLP, Deep Learning, TensorFlow, Python', 'Skill_Set_NLP, Python', 'Skill_Set_NLP, Python, Deep Learning', 'Skill_Set_NLP, Python, Deep Learning, TensorFlow', 'Skill_Set_NLP, Python, TensorFlow', 'Skill_Set_NLP, Python, TensorFlow, Deep Learning', 'Skill_Set_NLP, TensorFlow', 'Skill_Set_NLP, TensorFlow, Deep Learning', 'Skill_Set_NLP, TensorFlow, Deep Learning, Python', 'Skill_Set_NLP, TensorFlow, Python', 'Skill_Set_NLP, TensorFlow, Python, Deep Learning', 'Skill_Set_Negotiation, Communication', 'Skill_Set_Negotiation, Communication, Sales', 'Skill_Set_Negotiation, Sales', 'Skill_Set_Negotiation, Sales, Communication', 'Skill_Set_Operations Management, Excel', 'Skill_Set_Operations Management, Excel, Logistics', 'Skill_Set_Operations Management, Logistics', 'Skill_Set_Operations Management, Logistics, Excel', 'Job_ID', 'Talent_Inflow', 'Talent_Outflow', 'Infrastructure_Score', 'Smart_City_Investment', 'GCC_Presence', 'MSME_Growth_Rate', 'Unemployment_Rate', 'Education_Hubs', 'Cost_of_Living_Index', 'day', 'month', 'year', 'weekday', 'quarter']


# --- UI Layout ---
st.title("👨‍💻 Employee Salary Prediction")
st.markdown("This application predicts the salary range for a job in a non-metro Indian city. Fill in the details in the sidebar to get a prediction.")

# Display a warning if the models could not be loaded
if model is None or scaler is None:
    st.error(
        "**Failed to load model assets!** Please ensure the `final_model.pkl` and `scaler.pkl` files are present. "
        "You may need to run the `create_scaler.py` script first as described in the README."
    )
else:
    # --- Sidebar for User Inputs ---
    with st.sidebar:
        st.header("📋 Enter Job Details")

        # Use columns for a cleaner layout
        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("City", ['Ranchi', 'Vijayawada', 'Vadodara', 'Agra', 'Jodhpur', 'Visakhapatnam', 'Madurai', 'Raipur', 'Rajkot', 'Nashik'])
            job_role = st.selectbox("Job Role", ['Retail', 'AI/ML', 'Sales', 'Data Science', 'Engineering', 'Operations', 'Business Development', 'Others'])
        with col2:
            industry = st.selectbox("Industry", ['Retail', 'Tech', 'Manufacturing', 'Education', 'Finance', 'Pharma', 'Healthcare', 'Others'])
            experience_level = st.selectbox("Experience Level", ['Mid-Level', 'Entry-Level', 'Senior-Level', 'Internship'])

        # Skill set is a complex feature; for this app, we'll simplify it to a text input.
        # The user should enter a value that exists in the training data for best results.
        skill_set = st.text_input("Skill Set (e.g., Python, SQL)", 'Python, SQL, Data Visualization')
        
        st.subheader("📈 City & Economic Metrics")
        col3, col4 = st.columns(2)
        with col3:
            talent_inflow = st.number_input("Talent Inflow", min_value=0, value=1159)
            infra_score = st.slider("Infrastructure Score", min_value=0.0, max_value=100.0, value=74.25)
            msme_growth = st.number_input("MSME Growth Rate (%)", value=6.2)
            education_hubs = st.number_input("Education Hubs", min_value=0, value=29)
        with col4:
            talent_outflow = st.number_input("Talent Outflow", min_value=0, value=1583)
            smart_city_inv = st.number_input("Smart City Investment", min_value=0.0, value=9162.21)
            unemployment_rate = st.number_input("Unemployment Rate (%)", value=9.28)
            cost_of_living = st.slider("Cost of Living Index", min_value=0.0, max_value=120.0, value=74.65)

        st.subheader("⚙️ Other Details")
        col5, col6 = st.columns(2)
        with col5:
             gcc_presence = st.selectbox("GCC Presence", [0, 1], help="1 if a Global Capability Center is present, 0 if not")
             job_id = st.number_input("Job ID (simulation)", min_value=1, value=12001)
        with col6:
            job_posting_date = st.date_input("Job Posting Date", datetime.date(2024, 9, 24))


    # --- Prediction Logic ---
    if st.button("Predict Salary Range", type="primary", use_container_width=True):
        with st.spinner("🧠 Analyzing data..."):
            # Collect user input into a dictionary
            user_data = {
                'Job_ID': job_id, 'City': city, 'Industry': industry, 'Job_Role': job_role,
                'Skill_Set': skill_set, 'Experience_Level': experience_level,
                'Talent_Inflow': talent_inflow, 'Talent_Outflow': talent_outflow,
                'Infrastructure_Score': infra_score, 'Smart_City_Investment': smart_city_inv,
                'GCC_Presence': gcc_presence, 'MSME_Growth_Rate': msme_growth,
                'Unemployment_Rate': unemployment_rate, 'Education_Hubs': education_hubs,
                'Cost_of_Living_Index': cost_of_living, 'Job_Posting_Date': str(job_posting_date),
            }

            try:
                # Preprocess the input to get a model-ready DataFrame
                processed_input = preprocess_input(user_data, scaler, MODEL_COLUMNS)
                
                # Make prediction
                prediction = model.predict(processed_input)
                predicted_salary = prediction[0]

                # Display result in the main area
                st.success("Prediction Successful!")
                st.metric(
                    label="Predicted Annual Salary (INR)",
                    value=f"₹ {predicted_salary:,.2f}"
                )
                st.info("Note: This prediction is based on the provided model and data, and represents an estimated salary range.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please ensure all inputs are filled correctly. The 'Skill_Set' must be a comma-separated string that was present in the original training data for an accurate prediction.")

