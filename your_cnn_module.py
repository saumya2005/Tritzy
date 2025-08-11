from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# Load the trained MobileNetV2 model
model = load_model("saved_models/wheat_model.keras")

# Class labels (make sure they match your training order)
class_labels = ["Brown Rust", "Flag Smut", "Healthy", "Loose Smut", "Yellow Rust"]

def predict_disease_cnn(uploaded_file):
    contents = uploaded_file.file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))  # MobileNetV2 standard input size

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize if your model was trained this way

    prediction = model.predict(x)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    return {
        "predicted_class": predicted_class,
        "confidence": f"{confidence}%"
    }
