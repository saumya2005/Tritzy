from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from weather_module.weather_api import get_weather
from your_cnn_module import predict_disease_cnn
import joblib
import numpy as np
from get_proxy_weather import adjust_weather_with_proxies


app = FastAPI()

# Allow frontend requests (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Random Forest model
rf_model = joblib.load("weather_module/weather_model.pkl")

# Set your OpenWeather API key here
OPENWEATHER_API_KEY = "b993eadcc639b4ce0630be182f1eecb5"


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    city: str = Form(...),
    shade_level: str = Form(...),     # NEW: Proxy input from user
    irrigation: str = Form(...)       # NEW: Proxy input from user
):
    # Step 1: Run CNN prediction
    cnn_result = predict_disease_cnn(file)

    # Step 2: Get weather data from OpenWeather
    weather = get_weather(city, OPENWEATHER_API_KEY)
    if weather is None:
        return {"error": "Could not fetch weather data."}

    # Step 3: Adjust weather data with proxies
    proxy_inputs = {
        "shade_level": shade_level,
        "irrigation": irrigation
    }
    adjusted_weather = adjust_weather_with_proxies(weather, proxy_inputs)

    # Step 4: Prepare features for Random Forest model
    features = np.array([[adjusted_weather['temperature'],
                          adjusted_weather['humidity']]])

    # Step 5: Predict risk
    risk_prediction = rf_model.predict(features)[0]

    return {
        "cnn_prediction": cnn_result,
        "microclimate_risk":int(risk_prediction),
        "weather_details": adjusted_weather,  # Return adjusted data
        "original_weather": weather,          # Optional: show both
        "proxy_inputs": proxy_inputs          # Optional: for frontend display
    }
