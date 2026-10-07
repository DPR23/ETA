# Logistics ETA Predictor (LightGBM)

This repository contains a LightGBM machine learning model that predicts delivery arrival times based on route distance, traffic severity, weather conditions, and truck type. 

**Live Demo:** [Click here to test the model on Hugging Face]([YOUR_HUGGINGFACE_LINK](https://huggingface.co/spaces/DPR23/ETAPREDICT))



TO TEST LOCALLY






# Predictive ETA Routing API
1. Install requirements: `pip install -r requirements.txt`
2. Train the model: `python train.py`
3. Run the API: `uvicorn app:app --reload`
4. Test: Send a POST request to `http://127.0.0.1:8000/predict_eta` with JSON payload:
   {"distance_km": 150, "traffic_index": 4, "weather_condition": 1, "truck_type": 1}
