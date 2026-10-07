import gradio as gr
import lightgbm as lgb
import numpy as np
import joblib
import spaces

# 1. Load your trained model
model = joblib.load('lgbm_eta_model.pkl')

# 2. Prediction Function
@spaces.GPU
def predict_eta(distance, traffic, weather, truck):
    # Convert inputs into a 2D array. Order MUST match the training data!
    features = np.array([[distance, traffic, weather, truck]])
    
    # Predict
    prediction = model.predict(features)
    
    # Return formatted string
    return f"Estimated Delivery ETA: {round(prediction[0], 1)} minutes"

# 3. Gradio Interface
demo = gr.Interface(
    fn=predict_eta,
    title="Logistics ETA Predictor (LightGBM)",
    description="Predicts delivery arrival times based on route distance, traffic, weather, and truck type.",
    inputs=[
        gr.Number(label="Route Distance (km)", value=50),
        gr.Slider(minimum=1, maximum=10, step=1, label="Traffic Severity (1-10)"),
        gr.Dropdown(choices=[0, 1, 2], label="Weather Condition (0=Clear, 1=Rain, 2=Snow)", value=0),
        gr.Dropdown(choices=[0, 1], label="Truck Type (0=Standard, 1=Refrigerated)", value=0)
    ],
    outputs=gr.Text(label="Predicted ETA"),
)

demo.launch()
