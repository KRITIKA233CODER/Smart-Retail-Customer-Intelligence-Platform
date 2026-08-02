import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def decode_image(image_bytes: bytes):
    """Convert uploaded image bytes into an OpenCV image."""
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError("Invalid image")
    return image


def resize_image(image, width=224, height=224):
    return cv2.resize(image, (width, height))


def grayscale(image):
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def blur_image(image, kernel_size=5):
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


def detect_edges(image):
    gray = grayscale(image)
    edges = cv2.Canny(
        gray,
        100,
        200
    )
    return edges


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def detect_faces(image):
    gray = grayscale(image)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    results = []
    for (x, y, w, h) in faces:
        results.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        })
    return results


def get_embedding_extractor():
    """Load MobileNetV2 base model with average pooling for face embedding extraction."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    base_model.trainable = False
    return base_model


def extract_face_embedding(cropped_face, extractor_model):
    """Generate 1280-dimensional face embedding using MobileNetV2 extractor."""
    # Preprocess
    face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    face_batch = np.expand_dims(face_resized.astype(np.float32), axis=0)
    face_preprocessed = preprocess_input(face_batch)
    
    # Run extractor model
    embedding = extractor_model.predict(face_preprocessed, verbose=0)[0]
    # Normalize embedding vector
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.tolist()