import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "product_classifier.keras"
CLASSES_PATH = BASE_DIR / "models" / "product_classes.json"


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading product classifier...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Product classifier loaded.")


# --------------------------------------------------
# Load classes
# --------------------------------------------------

with open(CLASSES_PATH, "r") as file:
    class_mapping = json.load(file)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def classify_product(image):

    if image is None:
        raise ValueError("Invalid image")

    # OpenCV uses BGR; training images were RGB.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = image.astype(
        np.float32
    )

    # Add batch dimension:
    # (224,224,3) -> (1,224,224,3)
    image = np.expand_dims(
        image,
        axis=0
    )

    predictions = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index]
    )

    predicted_class = class_mapping[
        str(predicted_index)
    ]

    probabilities = {
        class_mapping[str(i)]: round(
            float(probability),
            6
        )
        for i, probability in enumerate(predictions)
    }

    return {
        "category": predicted_class,
        "confidence": round(
            confidence,
            6
        ),
        "confidence_percent": round(
            confidence * 100,
            2
        ),
        "probabilities": probabilities
    }