import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# Load Pickle Files
# =========================
base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, "final_model.pkl"))
rfe = joblib.load(os.path.join(base_path, "rfe.pkl"))
feature_names = joblib.load(os.path.join(base_path, "feature_names.pkl"))

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="💰 Salary Prediction App", layout="wide")
st.title("💰 Salary Prediction App")
st.write("Predict salaries using trained RandomForest model and selected features.")

# =========================
# Tabs: Single or Bulk Prediction
# =========================
tab1, tab2 = st.tabs(["Single Prediction", "Bulk Prediction"])

# -------------------------
# Single Prediction Tab
# -------------------------
with tab1:
    st.header("Predict for Single Employee")

    # Dynamic input fields
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Salary"):
        # Transform input using RFE
        transformed_input = rfe.transform(input_df)
        # Predict salary
        prediction = model.predict(transformed_input)
        st.success(f"💵 Predicted Salary: {prediction[0]:,.2f}")

# -------------------------
# Bulk Prediction Tab
# -------------------------
with tab2:
    st.header("Predict from CSV File")
    st.write("Upload a CSV file containing all features used in training.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Check if all required features are present
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            st.error(f"Missing features in CSV: {missing_features}")
        else:
            # Transform and predict
            transformed_data = rfe.transform(df[feature_names])
            predictions = model.predict(transformed_data)
            df["Predicted_Salary"] = predictions

            st.success("✅ Predictions completed!")
            st.dataframe(df)

            # Download predictions
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predicted_salaries.csv",
                mime="text/csv"
            )
