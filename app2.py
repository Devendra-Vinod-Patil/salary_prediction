# app.py
import os
import joblib
import pandas as pd
import streamlit as st

# attempt import of sklearn Pipeline class for some checks
try:
    from sklearn.pipeline import Pipeline
except Exception:
    Pipeline = None

BASE_DIR = os.path.dirname(__file__)

# -------------------------
# Load required .pkl files
# -------------------------
pipeline_path = os.path.join(BASE_DIR, "salary_pipeline.pkl")
model_path = os.path.join(BASE_DIR, "final_model.pkl")
features_path = os.path.join(BASE_DIR, "feature_names.pkl")
rfe_path = os.path.join(BASE_DIR, "rfe.pkl")

pipeline = joblib.load(pipeline_path)
final_model = joblib.load(model_path)
feature_names = joblib.load(features_path)        # expected: list-like of feature names
rfe = joblib.load(rfe_path)                       # expected: sklearn.feature_selection.RFE

# extract selected features from RFE object
try:
    selected_features = [f for f, s in zip(feature_names, rfe.support_) if s]
except Exception as e:
    # fallback: if rfe is already a list of feature names
    if isinstance(rfe, (list, tuple)):
        selected_features = list(rfe)
    else:
        raise RuntimeError("Could not extract selected features from rfe.pkl: " + str(e))

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Fill in details and click **Predict Salary**.")

# show selected features in sidebar for debugging
st.sidebar.header("Model info")
st.sidebar.write("Features used by model (RFE-selected):")
for f in selected_features:
    st.sidebar.write("- " + str(f))

st.subheader("Enter job details")

# --- Input fields (adjust these to match your dataset's feature names) ---
# The fields below are the same you provided earlier. If your selected_features
# contains different names, you'll see a helpful error asking to add those inputs.
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

experience = st.slider("Years of Experience", 0, 30, 1)
education_score = st.number_input("Education Hubs (count)", min_value=0, step=1, value=0)
infra_score = st.number_input("Infrastructure Score", min_value=0.0, step=0.1, value=0.0)

# Build input dataframe with same column names as used during training (modify if needed)
input_row = {
    "City": city,
    "Industry": industry,
    "Job_Role": job_role,
    "Skills": ", ".join(skills),
    "Experience": experience,
    "Education_Hubs": education_score,
    "Infrastructure_Score": infra_score
}

input_df = pd.DataFrame([input_row])

# Keep only columns that are part of the model's selected features
required_cols = [c for c in selected_features]
missing_cols = [c for c in required_cols if c not in input_df.columns]

if missing_cols:
    st.warning(
        "The app UI does not provide inputs for all RFE-selected features. "
        "Missing features: " + ", ".join(missing_cols)
    )
    st.info("Either update this app to collect those inputs or re-run training/pickle with features matching this UI.")
    # Stop before attempting prediction because pipeline will likely fail
    st.stop()

# reorder columns as model expects
df_for_model = input_df[required_cols]

# -------------------------
# Prediction logic (robust)
# -------------------------
def try_transform_then_predict(preproc, model, X):
    """Try to transform X with preproc, then predict with model. Returns (pred, method_used) or (None, reason)."""
    if hasattr(preproc, "transform"):
        X_trans = preproc.transform(X)
        if hasattr(model, "predict"):
            return model.predict(X_trans)[0], "preproc.transform -> final_model.predict"
        else:
            return None, "final_model has no predict method"
    return None, "preprocessor has no transform"

def try_pipeline_predict(full_pipeline, X):
    """If pipeline already contains estimator, call predict directly."""
    if hasattr(full_pipeline, "predict"):
        return full_pipeline.predict(X)[0], "pipeline.predict (pipeline contains estimator)"
    return None, "pipeline has no predict"

def try_strip_last_step_and_transform(full_pipeline, model, X):
    """If sklearn Pipeline, remove last step and try transform -> predict"""
    if Pipeline is None:
        return None, "sklearn Pipeline unavailable"
    if isinstance(full_pipeline, Pipeline) and len(full_pipeline.steps) > 1:
        preproc = Pipeline(full_pipeline.steps[:-1])
        if hasattr(preproc, "transform") and hasattr(model, "predict"):
            X_t = preproc.transform(X)
            return model.predict(X_t)[0], "Pipeline[:-1].transform -> final_model.predict"
        return None, "stripped pipeline lacks transform or final_model lacks predict"
    return None, "not an sklearn Pipeline or only one step"

def try_named_preprocessor(full_pipeline, model, X):
    """Try common named_steps keys like 'preprocessor' or 'transformer'."""
    named = getattr(full_pipeline, "named_steps", None)
    if not named:
        return None, "no named_steps"
    for key in ("preprocessor", "transformer", "scaler", "encoder"):
        obj = named.get(key)
        if obj is not None and hasattr(obj, "transform") and hasattr(model, "predict"):
            X_t = obj.transform(X)
            return model.predict(X_t)[0], f"named_steps['{key}'].transform -> final_model.predict"
    return None, "no suitable named preprocessor found"

# Main predict action
if st.button("Predict Salary"):
    try:
        pred = None
        method = None

        # 1) If pipeline has transform, prefer that and then final_model.predict
        if hasattr(pipeline, "transform"):
            try:
                pred, method = try_transform_then_predict(pipeline, final_model, df_for_model)
            except Exception as e:
                pred = None
                method = f"pipeline.transform failed: {e}"

        # 2) If pipeline itself can predict, use that directly
        if pred is None and hasattr(pipeline, "predict"):
            try:
                pred, m = try_pipeline_predict(pipeline, df_for_model)
                method = method or m
            except Exception as e:
                pred = None
                method = (method or "") + f" ; pipeline.predict failed: {e}"

        # 3) If pipeline is sklearn Pipeline, strip last step (assumed estimator) and transform
        if pred is None:
            try:
                pred, m = try_strip_last_step_and_transform(pipeline, final_model, df_for_model)
                method = method or m
            except Exception as e:
                pred = None
                method = (method or "") + f" ; strip-last failed: {e}"

        # 4) Try common named_steps (preprocessor, transformer, etc.)
        if pred is None:
            try:
                pred, m = try_named_preprocessor(pipeline, final_model, df_for_model)
                method = method or m
            except Exception as e:
                pred = None
                method = (method or "") + f" ; named_steps check failed: {e}"

        # 5) As a final fallback, if final_model itself accepts raw df (rare), try it
        if pred is None and hasattr(final_model, "predict"):
            try:
                pred = final_model.predict(df_for_model)[0]
                method = method or "final_model.predict on raw inputs (fallback)"
            except Exception as e:
                pred = None
                method = (method or "") + f" ; final_model.predict(raw) failed: {e}"

        if pred is None:
            st.error("Could not obtain a prediction. Details: " + str(method))
            st.stop()

        st.success(f"💰 Predicted Salary: ₹{pred:,.2f}")
        st.info(f"Used method: {method}")

    except Exception as e:
        st.error("Prediction failed: " + str(e))
