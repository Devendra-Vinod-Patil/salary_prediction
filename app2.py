import gradio as gr
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
# Prediction Function
# =========================
def predict_salary(*inputs):
    """
    inputs : values in the same order as feature_names
    """
    input_df = pd.DataFrame([inputs], columns=feature_names)
    transformed = rfe.transform(input_df)
    prediction = model.predict(transformed)
    return round(prediction[0], 2)

# =========================
# Gradio Interface
# =========================
input_widgets = []

# Create input widgets dynamically based on feature names
for feature in feature_names:
    input_widgets.append(gr.Number(label=feature, value=0))

iface = gr.Interface(
    fn=predict_salary,
    inputs=input_widgets,
    outputs=gr.Textbox(label="💵 Predicted Salary"),
    title="💰 Employee Salary Prediction",
    description="Enter employee information below to predict the salary using trained RandomForest model."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
