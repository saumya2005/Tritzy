# predict_combined.py

import joblib
from get_proxy_weather import get_proxy_weather
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# --------------------
# Step 1: Weather Prediction
# --------------------
# Load trained Random Forest model
weather_model = joblib.load('weather_module/weather_model.pkl')

# Ask user for field condition
print("Select field condition (Wet and Humid / Dry and Windy / Cool and Damp / Hot and Dry):")
condition = input("Enter condition: ")
features = get_proxy_weather(condition)
input_features = [[features['temperature'], features['humidity']]]

weather_prediction = weather_model.predict(input_features)[0]

# --------------------
# Step 2: CNN Image Prediction
# --------------------
cnn_model = load_model('saved_models/wheat_model.keras')

img_path = 'wheat_dataset/test/healthy_test/healthy_test_3.png'  # Update this as needed
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

preds = cnn_model.predict(img_array)
cnn_prediction = np.argmax(preds, axis=1)[0]  # scalar

# Label mapping
label_map = {
    0: "Flag Smut",
    1: "Loose Smut",
    2: "Brown Rust",
    3: "Yellow Rust",
    4: "Healthy"
}

# --------------------
# Step 3: Output Final Prediction
# --------------------
print("\n--- Predictions ---")
print("Weather model prediction:", weather_prediction, "-", label_map[weather_prediction])
print("CNN predicted:", cnn_prediction, "-", label_map[cnn_prediction])

if weather_prediction == cnn_prediction:
    print(f"✅ Both models agree. Final prediction: {label_map[cnn_prediction]}")
else:
    print(f"⚠️ Models disagree.\nSuggested Output: Image-based - {label_map[cnn_prediction]}, Weather-based - {label_map[weather_prediction]}")
