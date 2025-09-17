import gradio as gr
import pandas as pd
import joblib
import os

# =========================
# Load trained artifacts
# =========================
base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, "final_model.pkl"))
rfe = joblib.load(os.path.join(base_path, "rfe.pkl"))
feature_names = joblib.load(os.path.join(base_path, "feature_names.pkl"))
encoders = joblib.load(os.path.join(base_path, "encoders.pkl"))  # dict of LabelEncoders/OneHot for categorical features

# =========================
# Prediction Function
# =========================
def predict_salary(experience, state, company, role, skills):
    """
    experience: float
    state: str
    company: str
    role: str
    skills: list of strings
    """
    # Build dataframe with all features
    input_dict = {}

    # Numeric features
    input_dict['experience'] = [experience]

    # Encode categorical features using saved encoders
    input_dict['state'] = [encoders['state'].transform([state])[0]]
    input_dict['company'] = [encoders['company'].transform([company])[0]]
    input_dict['role'] = [encoders['role'].transform([role])[0]]

    # Skills: multi-hot encoding
    all_skills = encoders['skills'].classes_  # all possible skills
    skill_vector = [1 if s in skills else 0 for s in all_skills]
    for i, s in enumerate(all_skills):
        input_dict[f"skill_{s}"] = [skill_vector[i]]

    # Convert to DataFrame
    input_df = pd.DataFrame(input_dict)

    # Select only RFE features
    input_df_rfe = input_df[feature_names]
    transformed = rfe.transform(input_df_rfe)
    prediction = model.predict(transformed)
    return round(prediction[0], 2)

# =========================
# Gradio Interface
# =========================
experience_input = gr.Slider(0, 40, step=1, label="Experience (years)")
state_input = gr.Dropdown(encoders['state'].classes_.tolist(), label="State")
company_input = gr.Dropdown(encoders['company'].classes_.tolist(), label="Company")
role_input = gr.Dropdown(encoders['role'].classes_.tolist(), label="Role")
skills_input = gr.CheckboxGroup(encoders['skills'].classes_.tolist(), label="Skills")

iface = gr.Interface(
    fn=predict_salary,
    inputs=[experience_input, state_input, company_input, role_input, skills_input],
    outputs=gr.Textbox(label="💵 Predicted Salary"),
    title="💰 Employee Salary Prediction",
    description="Enter employee details to predict realistic salary based on skills, experience, location, role, and company."
)

if __name__ == "__main__":
    iface.launch()
