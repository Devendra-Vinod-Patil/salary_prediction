# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="💼 Salary Prediction (Robust)", layout="wide")

@st.cache_resource
def load_model(path="final_model.pkl"):
    model = joblib.load(path)
    return model

def is_pipeline(m):
    return isinstance(m, Pipeline) or hasattr(m, "named_steps")

def get_feature_names_from_model(model):
    # Prefer model.feature_names_in_ (final estimator or pre-fitted transformer)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # If pipeline, try last estimator
    if is_pipeline(model):
        try:
            last = list(model.named_steps.values())[-1]
            if hasattr(last, "feature_names_in_"):
                return list(last.feature_names_in_)
        except Exception:
            pass
    return None

def extract_columns_from_preprocessor(preprocessor):
    """
    If the pipeline contains a ColumnTransformer-like step, return the list
    of input column names used by the preprocessor.
    """
    cols = []
    try:
        if isinstance(preprocessor, ColumnTransformer) or hasattr(preprocessor, "transformers_"):
            for name, transformer, selector in preprocessor.transformers_:
                if selector == "drop":
                    continue
                if isinstance(selector, (list, tuple)):
                    cols.extend(list(selector))
                elif isinstance(selector, str):
                    # `'remainder'` or column name
                    if selector.lower() not in ("remainder", "drop"):
                        cols.append(selector)
                else:
                    # could be slice or boolean mask; skip
                    pass
    except Exception:
        pass
    # dedupe while preserving order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def sanitize_feature_name(name):
    return re.sub(r'[^a-z0-9]', '_', name.lower())

def find_best_skill_feature(skill_features, selected_skills):
    """
    Given model skill feature names (like 'Skill_Set_Python,_SQL,_Project_Management')
    and a list of selected skills (['Python','SQL']), find best matching feature by token overlap.
    """
    if not selected_skills:
        return None
    best = None
    best_score = -1
    input_tokens = set([sanitize_feature_name(s) for s in selected_skills])
    for feat in skill_features:
        # remove prefix "Skill_Set_" if present
        token_part = feat
        if feat.lower().startswith("skill_set_"):
            token_part = feat[len("skill_set_"):]
        # split by non-alphanum
        feat_tokens = set([t for t in re.split(r'[^a-z0-9]+', token_part.lower()) if t])
        score = len(feat_tokens & input_tokens)
        # also prefer smaller difference in size
        if score > best_score or (score == best_score and best is not None and len(feat_tokens) < len(re.split(r'[^a-z0-9]+', best[len("Skill_Set_"):].lower()))):
            best_score = score
            best = feat
    return best if best_score > 0 else None

def build_one_hot_input_from_features(model_features, city, industry, job_role, exp_level, company, selected_skills, cost_index, unemployment_rate, infra_score):
    # initialize zeros
    X = {f: 0 for f in model_features}
    # categorical one-hot mapping
    mapping = {
        "City": city,
        "Industry": industry,
        "Job_Role": job_role,
        "Experience_Level": exp_level,
        "Company_Name": company
    }
    for prefix, val in mapping.items():
        if val is None:
            continue
        colname = f"{prefix}_{val}"
        # some models might use spaces or slightly different capitalization; try variations
        if colname in X:
            X[colname] = 1
        else:
            # try sanitized comparison
            for f in model_features:
                if sanitize_feature_name(f) == sanitize_feature_name(colname):
                    X[f] = 1
                    break

    # Skills: find best matching Skill_Set_* column from model_features
    skill_features = [f for f in model_features if f.startswith("Skill_Set_")]
    best_skill_col = find_best_skill_feature(skill_features, selected_skills)
    if best_skill_col:
        X[best_skill_col] = 1

    # numeric features - set if present
    for num_col, value in [
        ("Cost_of_Living_Index", cost_index),
        ("Unemployment_Rate", unemployment_rate),
        ("Infrastructure_Score", infra_score)
    ]:
        if num_col in X:
            X[num_col] = value
        else:
            # try sanitized match
            for f in model_features:
                if sanitize_feature_name(f) == sanitize_feature_name(num_col):
                    X[f] = value
                    break

    # final DataFrame (correct order)
    X_df = pd.DataFrame([[X[f] for f in model_features]], columns=model_features)
    return X_df

def map_ui_to_raw_columns(expected_cols, city, industry, job_role, exp_level, company, selected_skills, cost_index, unemployment_rate, infra_score):
    """Attempt to build a raw DataFrame matching expected_cols names by heuristic mapping."""
    raw = {}
    for c in expected_cols:
        lc = c.lower()
        if "city" in lc and ("city_" not in lc and not lc.startswith("city_")):
            # fill with simple city string
            raw[c] = city
            continue
        if "industry" in lc:
            raw[c] = industry
            continue
        if "job" in lc or "role" in lc:
            raw[c] = job_role
            continue
        if "experience" in lc:
            raw[c] = exp_level
            continue
        if "company" in lc:
            raw[c] = company
            continue
        if "skill" in lc:
            # If the model expects a raw column named like 'Skill_Set' or 'skill_set',
            # set a combined string of skills (comma separated).
            raw[c] = ", ".join(selected_skills) if selected_skills else ""
            continue
        if sanitize_feature_name(c) in ("cost_of_living_index", "cost_of_living", "costindex"):
            raw[c] = cost_index
            continue
        if "unemployment" in lc:
            raw[c] = unemployment_rate
            continue
        if "infrastructure" in lc or "infra" in lc:
            raw[c] = infra_score
            continue
        # default NaN for unknown
        raw[c] = np.nan
    return pd.DataFrame([raw])

# ----------------------
# UI
# ----------------------
st.title("💼 Salary Prediction (Robust)")

st.markdown("Use the dropdowns and multiselect. The app will try to choose the correct preprocessing path automatically.")

# UI choices (kept small and relevant)
cities = ["Madurai", "Nashik", "Raipur", "Rajkot", "Ranchi", "Vadodara", "Vijayawada", "Visakhapatnam"]
industries = ["Education", "Finance", "Healthcare", "Manufacturing", "Pharma", "Retail", "Tech", "Others"]
job_roles = ["AI/ML", "Business Development", "Data Science", "Engineering", "Operations", "Retail", "Sales", "Others"]
exp_levels = ["Internship", "Entry-Level", "Mid-Level", "Senior-Level", "Lead/Manager"]
companies = ["HCLTech", "Reliance Retail", "Others"]

base_skills = [
    "Python", "Java", "SQL", "Data Analysis", "Machine Learning", "Deep Learning", "NLP", "TensorFlow",
    "Project Management", "Agile", "Scrum", "Sales", "CRM", "Market Analysis", "Customer Service", "MS Office",
    "Communication", "Negotiation", "Logistics", "Operations Management", "Excel", "Statistics", "Data Visualization"
]

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", cities)
    industry = st.selectbox("Industry", industries)
    job_role = st.selectbox("Job Role", job_roles)
    exp_level = st.selectbox("Experience Level", exp_levels)
    company = st.selectbox("Company", companies)

with col2:
    selected_skills = st.multiselect("Skills (choose as many as apply)", base_skills)
    cost_index = st.number_input("Cost of Living Index", min_value=1.0, max_value=1000.0, value=80.0, step=0.1)
    unemployment_rate = st.slider("Unemployment Rate (%)", min_value=0.0, max_value=50.0, value=5.0)
    infra_score = st.slider("Infrastructure Score (0-100)", min_value=0.0, max_value=100.0, value=50.0)

st.markdown("---")

model = None
try:
    model = load_model("final_model.pkl")
    st.success("Loaded 'final_model.pkl' successfully.")
except Exception as e:
    st.error(f"Failed to load 'final_model.pkl': {e}")
    st.stop()

st.markdown("**Model inspection:**")
st.write(type(model))

# Attempt to find model features
MODEL_FEATURES = get_feature_names_from_model(model)
st.write("Detected model.feature_names_in_ ?: ", MODEL_FEATURES is not None)

# If the pipeline has a ColumnTransformer/ preprocessor, try to extract expected raw columns
preprocessor_cols = None
if is_pipeline(model):
    # find a ColumnTransformer in named_steps
    pre = None
    for name, step in model.named_steps.items():
        if isinstance(step, ColumnTransformer) or hasattr(step, "transformers_"):
            pre = step
            break
    if pre is not None:
        pre_cols = extract_columns_from_preprocessor(pre)
        if pre_cols:
            preprocessor_cols = pre_cols
            st.write("Extracted preprocessor input columns (heuristic):")
            st.write(preprocessor_cols)

# Skill_set features in model features (if available)
skill_set_features = []
if MODEL_FEATURES is not None:
    skill_set_features = [f for f in MODEL_FEATURES if f.startswith("Skill_Set_")]
    if skill_set_features:
        st.write(f"Model has {len(skill_set_features)} combined Skill_Set features (examples): {skill_set_features[:10]}")

# MAIN PREDICT BUTTON
if st.button("Predict Salary"):
    # Clip numeric inputs to reasonable ranges (avoids extreme values)
    cost_index = float(np.clip(cost_index, 1.0, 1000.0))
    unemployment_rate = float(np.clip(unemployment_rate, 0.0, 50.0))
    infra_score = float(np.clip(infra_score, 0.0, 100.0))

    prediction = None
    debug_info = {}

    # 1) First try: If model is a pipeline, pass a RAW DataFrame with preprocessor_cols (best)
    if is_pipeline(model) and preprocessor_cols:
        st.write("Attempting pipeline prediction using preprocessor columns...")
        raw_df = map_ui_to_raw_columns(preprocessor_cols, city, industry, job_role, exp_level, company, selected_skills, cost_index, unemployment_rate, infra_score)
        debug_info['path'] = 'pipeline_preprocessor_cols'
        debug_info['raw_df_head'] = raw_df.head().to_dict()
        try:
            pred = model.predict(raw_df)
            prediction = float(pred[0])
            debug_info['note'] = "Succeeded with pipeline preprocessor columns."
        except Exception as e:
            debug_info['pipeline_error'] = str(e)
            st.warning("Pipeline prediction with preprocessor columns failed. Will try alternative methods.")
    # 2) Second try: If model is pipeline but no preprocessor cols or first attempt failed,
    #    try passing a simple raw DataFrame with canonical column names (heuristic)
    if prediction is None and is_pipeline(model):
        st.write("Attempting pipeline prediction using heuristic raw columns...")
        # heuristic raw feature list (names the pipeline might expect)
        heuristic_raw_cols = ["City","Industry","Job_Role","Experience_Level","Company_Name","Skill_Set",
                              "Cost_of_Living_Index","Unemployment_Rate","Infrastructure_Score","month","day","year"]
        raw_df = map_ui_to_raw_columns(heuristic_raw_cols, city, industry, job_role, exp_level, company, selected_skills, cost_index, unemployment_rate, infra_score)
        debug_info['path'] = debug_info.get('path','') + ' | pipeline_heuristic_cols'
        debug_info['raw_df_head_heuristic'] = raw_df.head().to_dict()
        try:
            pred = model.predict(raw_df)
            prediction = float(pred[0])
            debug_info['note'] = debug_info.get('note','') + " Succeeded with pipeline heuristic cols."
        except Exception as e:
            debug_info['pipeline_heuristic_error'] = str(e)

    # 3) Third try: If the model exposes feature_names_in_ (i.e. expects a specific one-hot ordered input),
    #    construct that DataFrame exactly and call predict on it.
    if prediction is None and MODEL_FEATURES is not None:
        st.write("Attempting one-hot aligned prediction using model.feature_names_in_ ...")
        X_input = build_one_hot_input_from_features(MODEL_FEATURES, city, industry, job_role, exp_level, company, selected_skills, cost_index, unemployment_rate, infra_score)
        debug_info['path'] = debug_info.get('path','') + ' | one_hot_aligned'
        debug_info['X_input_head'] = X_input.head().to_dict()
        # diagnostics: missing/extra sets (should be none because we built from model features)
        try:
            pred = model.predict(X_input)
            prediction = float(pred[0])
            debug_info['note'] = debug_info.get('note','') + " Succeeded with one-hot aligned input."
        except Exception as e:
            debug_info['one_hot_error'] = str(e)

    # 4) Still None -> try naive attempt: if model is not pipeline and no feature names, try passing a tiny numeric array
    if prediction is None and not is_pipeline(model) and MODEL_FEATURES is None:
        st.write("As a last resort, trying to call predict with a single-row numeric array (not recommended).")
        try:
            pred = model.predict(np.zeros((1,1)))
            prediction = float(pred[0])
            debug_info['note'] = "Last-resort numeric predict succeeded (likely incorrect)."
        except Exception as e:
            debug_info['last_resort_error'] = str(e)

    # SHOW DEBUG INFO
    st.markdown("### Debug info (useful if prediction looks wrong)")
    st.json(debug_info)

    # Show prediction or error
    if prediction is not None:
        # Post-check: if prediction is implausible, warn and show raw value
        if not np.isfinite(prediction):
            st.error(f"Model returned non-finite prediction: {prediction}")
        else:
            if prediction < 0 or prediction > 1e7:  # > 10 million annual is suspicious for this dataset
                st.warning(f"Model returned an unusual value: ₹ {prediction:,.2f}. This may indicate preprocessing mismatch or scaling missing.")
                # still display but clearly
                st.markdown(f"<div style='background:#fff3cd;padding:12px;border-left:6px solid #ffc107;border-radius:6px;'>"
                            f"<strong>Predicted (raw):</strong> ₹ {prediction:,.2f}</div>", unsafe_allow_html=True)
            else:
                st.success(f"Predicted Annual Salary: ₹ {prediction:,.2f}")
    else:
        st.error("Failed to produce a prediction. See debug info above. Common causes:\n"
                 "- The saved model is not a pipeline and preprocessing (scaler/encoder) was not saved with it.\n"
                 "- Feature names/order in the app do not match exactly what the model expects.\n"
                 "Recommended actions:\n"
                 "1. If you trained the model, save the full sklearn Pipeline (preprocessor + estimator) via joblib.dump(pipeline, 'final_model.pkl').\n"
                 "2. If you only saved the estimator, also save the preprocessor (ColumnTransformer) and scaler and load them here before predict.\n"
                 "3. Inspect `model.feature_names_in_` and compare with the inputs printed above.")
