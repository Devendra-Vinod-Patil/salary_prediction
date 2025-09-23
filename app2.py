# app.py
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from typing import List

# -------------------------
# Helpers
# -------------------------
def load_pkls(base_dir: str):
    final_model = joblib.load(os.path.join(base_dir, "final_model.pkl"))
    feature_names = joblib.load(os.path.join(base_dir, "feature_names.pkl"))
    rfe_obj = joblib.load(os.path.join(base_dir, "rfe.pkl"))
    return final_model, feature_names, rfe_obj

def extract_selected_features(feature_names: List[str], rfe_obj):
    # rfe_obj may be an sklearn RFE or a list of feature names
    try:
        # sklearn RFE: has attribute support_
        mask = list(rfe_obj.support_)
        selected = [f for f, s in zip(feature_names, mask) if s]
    except Exception:
        if isinstance(rfe_obj, (list, tuple)):
            selected = list(rfe_obj)
        else:
            # fallback: if rfe_obj has .get_support()
            try:
                mask = list(rfe_obj.get_support())
                selected = [f for f, s in zip(feature_names, mask) if s]
            except Exception as e:
                raise RuntimeError("Unable to extract selected features from rfe.pkl: " + str(e))
    return selected

def is_probably_categorical(name: str) -> bool:
    # heuristics based on common tokens
    name_l = name.lower()
    cat_tokens = ["city", "state", "region", "industry", "role", "job", "dept",
                  "gender", "edu", "education", "degree", "qual", "category", "type"]
    for t in cat_tokens:
        if t in name_l:
            return True
    # names with limited distinct values implied by typical encodings
    if name_l.endswith("_cat") or name_l.endswith("_catg"):
        return True
    return False

def make_default_options(name: str):
    # reasonable default dropdown options for common categorical features
    name_l = name.lower()
    if "city" in name_l:
        return ["Mumbai","Delhi","Bengaluru","Hyderabad","Chennai","Pune","Kolkata","Other"]
    if "industry" in name_l:
        return ["Tech","Finance","Healthcare","Education","Manufacturing","Retail","Other"]
    if "role" in name_l or "job" in name_l:
        return ["Software Engineer","Data Scientist","Manager","Sales","Operations","Other"]
    if "gender" in name_l:
        return ["Male","Female","Other"]
    if "education" in name_l or "degree" in name_l or "edu" in name_l:
        return ["High School","Diploma","Graduate","Postgraduate","PhD","Other"]
    # generic
    return ["Option A","Option B","Option C"]

def build_input_widget(col_name: str):
    """Return the (label, value) pair for a single feature using heuristics."""
    if is_probably_categorical(col_name):
        options = make_default_options(col_name)
        value = st.selectbox(col_name, options, key=f"sel_{col_name}")
    else:
        # numeric: try to guess range from name tokens
        name_l = col_name.lower()
        if "experience" in name_l or "years" in name_l:
            value = st.slider(col_name, 0, 40, 2, key=f"num_{col_name}")
        elif "score" in name_l or "count" in name_l or "hubs" in name_l or "education_hubs" in name_l:
            value = st.number_input(col_name, min_value=0.0, step=1.0, value=0.0, key=f"num_{col_name}")
        else:
            # generic numeric
            value = st.number_input(col_name, value=0.0, step=0.1, key=f"num_{col_name}")
    return value

def compute_linear_contribs(model, feature_vector: np.ndarray, feature_names: List[str]):
    """
    If model is linear (has coef_), compute feature*coef contributions.
    Returns dict feature->contribution and intercept.
    """
    try:
        coef = np.array(model.coef_).ravel()
        if coef.size != feature_vector.shape[1]:
            # shapes mismatch
            return None, "coef size mismatch"
        contribs = coef * feature_vector.ravel()
        intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0
        return dict(zip(feature_names, contribs.tolist())), intercept
    except Exception as e:
        return None, str(e)

# -------------------------
# App start
# -------------------------
BASE_DIR = os.path.dirname(__file__)

st.set_page_config(page_title="Salary Predictor (no-pipeline)", layout="wide", page_icon="💼")
st.header("💼 Salary Prediction — Advanced UI (no pipeline)")

# Load model and metadata
with st.spinner("Loading model and metadata..."):
    try:
        model, feature_names_all, rfe_obj = load_pkls(BASE_DIR)
    except Exception as e:
        st.error(f"Could not load required .pkl files from {BASE_DIR}: {e}")
        st.stop()

# Derive selected features (these are expected to be original feature names used during training)
try:
    selected_features = extract_selected_features(feature_names_all, rfe_obj)
except Exception as e:
    st.error("Failed to determine RFE-selected features: " + str(e))
    st.stop()

# Sidebar: model info, presets, history
st.sidebar.title("Model & App")
st.sidebar.subheader("Files loaded")
st.sidebar.write("- final_model.pkl")
st.sidebar.write("- feature_names.pkl")
st.sidebar.write("- rfe.pkl")

st.sidebar.subheader("Selected features (RFE)")
for f in selected_features:
    st.sidebar.write("- " + f)

# Presets
st.sidebar.subheader("Presets")
preset = st.sidebar.selectbox("Choose sample profile", ["None", "Junior Data Scientist", "Experienced Manager", "Fresh Graduate"])
# history storage
if "history" not in st.session_state:
    st.session_state.history = []

# Layout: left column for inputs, right for results / diagnostics
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Input features")
    st.write("Provide values for the features below. App builds controls automatically from RFE-selected feature names.")
    # Use a form so the entire input is submitted together
    with st.form(key="predict_form"):
        input_values = {}
        # If user selected a preset, predefine some heuristic defaults
        for feat in selected_features:
            # apply preset defaults
            if preset == "Junior Data Scientist":
                if "experience" in feat.lower():
                    default_widget = 1
                elif "role" in feat.lower() or "job" in feat.lower():
                    default_widget = "Data Scientist"
                else:
                    default_widget = None
            elif preset == "Experienced Manager":
                if "experience" in feat.lower():
                    default_widget = 12
                elif "role" in feat.lower() or "job" in feat.lower():
                    default_widget = "Manager"
                else:
                    default_widget = None
            elif preset == "Fresh Graduate":
                if "experience" in feat.lower():
                    default_widget = 0
                elif "education" in feat.lower() or "degree" in feat.lower():
                    default_widget = "Graduate"
                else:
                    default_widget = None
            else:
                default_widget = None

            # create column label that includes short tooltip if needed
            label = feat
            # Create widget; attempt to set default if possible
            if is_probably_categorical(feat):
                options = make_default_options(feat)
                # if the default is present in options, use it
                if default_widget is not None and default_widget in options:
                    val = st.selectbox(label + " ✳", options, index=options.index(default_widget), key=f"widget_{feat}")
                else:
                    val = st.selectbox(label, options, key=f"widget_{feat}")
            else:
                # numeric
                if isinstance(default_widget, (int, float)):
                    val = st.number_input(label, value=float(default_widget), key=f"widget_{feat}")
                else:
                    # choose wide-range slider for experience, else number input
                    if "experience" in feat.lower() or "years" in feat.lower():
                        val = st.slider(label, 0, 40, 1, key=f"widget_{feat}")
                    else:
                        val = st.number_input(label, value=0.0, step=0.1, key=f"widget_{feat}")
            input_values[feat] = val

        submit = st.form_submit_button("Predict")

with right_col:
    st.subheader("Prediction & Diagnostics")
    st.write("Model type: `{}`".format(type(model).__name__))

    if submit:
        # Convert inputs into DataFrame / array expected by model
        try:
            # Build a DataFrame with columns ordered as selected_features
            df_input = pd.DataFrame([input_values], columns=selected_features)
            st.write("### Input summary")
            st.table(df_input.T.rename(columns={0: "value"}))

            # Attempt prediction - model may accept DataFrame or numpy array of shape (1,n)
            prediction = None
            details = ""

            # 1) try direct predict on DataFrame
            try:
                prediction = model.predict(df_input)[0]
                details = "model.predict(df_input) used"
            except Exception as e:
                details = f"model.predict(df_input) failed: {e}"

            # 2) try predict on numpy array (ordered)
            if prediction is None:
                try:
                    arr = df_input.to_numpy(dtype=float)
                except Exception:
                    # try fallback: convert categories to indices by hashing (best-effort)
                    try:
                        arr = df_input.apply(lambda col: pd.to_numeric(col, errors='coerce').fillna(0)).to_numpy(dtype=float)
                    except Exception as e:
                        st.error("Failed to prepare numeric input array: " + str(e))
                        st.stop()
                try:
                    prediction = model.predict(arr)[0]
                    details = "model.predict(numpy_array) used"
                except Exception as e:
                    details = (details or "") + f" ; model.predict(array) failed: {e}"

            # If still None, show helpful guidance
            if prediction is None:
                st.error("Prediction failed. Likely the model expects preprocessed/encoded inputs (a pipeline was used during training)."
                         " This app purposely avoids the pipeline. To predict successfully provide a model trained on raw selected features or supply the pipeline.")
                st.info("Technical details: " + details)
            else:
                # show prediction
                st.metric("Predicted Salary (INR)", f"₹{prediction:,.2f}")
                st.success("Prediction successful.")
                st.write("**Method:**", details)

                # Save to history
                hist_entry = df_input.copy()
                hist_entry["prediction"] = prediction
                st.session_state.history.append(hist_entry)

                # Attempt to show feature contributions when possible (linear models)
                contribs, c_info = compute_linear_contribs(model, df_input.to_numpy().reshape(1, -1), selected_features)
                if contribs:
                    st.subheader("Feature contributions (linear model)")
                    contrib_df = pd.DataFrame.from_dict(contribs, orient="index", columns=["contribution"]).sort_values("contribution", ascending=False)
                    contrib_df["abs"] = contrib_df["contribution"].abs()
                    contrib_df = contrib_df.sort_values("abs", ascending=False).drop(columns="abs")
                    st.bar_chart(contrib_df)
                    st.table(contrib_df)
                else:
                    st.info("Feature contributions unavailable: " + c_info)

                # Download button for single result
                csv_down = df_input.copy()
                csv_down["prediction"] = prediction
                csv_bytes = csv_down.to_csv(index=False).encode('utf-8')
                st.download_button("Download result (CSV)", data=csv_bytes, file_name="salary_prediction.csv", mime="text/csv")

        except Exception as e:
            st.error("An unexpected error occurred preparing or predicting: " + str(e))

    else:
        st.info("Fill inputs and click Predict to run the model.")

# Bottom: show history table (if any)
st.markdown("---")
st.subheader("Prediction history")
if len(st.session_state.history) == 0:
    st.write("No predictions yet.")
else:
    hist_df = pd.concat(st.session_state.history, ignore_index=True)
    st.dataframe(hist_df)

# Tips and advanced info
with st.expander("Tips & Notes (click to expand)"):
    st.markdown(
        """
        - This app purposely **does not use the preprocessing pipeline**. If your trained `final_model.pkl` expects encoded/scaled features,
          prediction will likely fail or give incorrect results. In that case you should either:
          1. Use the original `salary_pipeline.pkl` to transform raw inputs the same way as training, OR
          2. Re-train `final_model.pkl` on the raw features you provide here (so it accepts the same feature columns).
        - The app auto-generates input widgets using heuristics on feature names. If a feature is mis-typed or needs a different control type,
          edit the app to customize that widget.
        - If you want, I can update the app to:
            * Auto-extract categorical levels from a saved encoder inside a pipeline (if you allow loading the pipeline), OR
            * Auto-create one-hot encoded columns to exactly match `feature_names.pkl` (requires info about how categories were encoded).
        """
    )

