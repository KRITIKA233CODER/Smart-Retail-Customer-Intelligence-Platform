import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app

API_KEY_HEADERS = {"X-API-Key": "smart-retail-secret-key"}

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# ============================================================
# Basic API
# ============================================================

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "message" in data


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ============================================================
# Authentication Security Checks
# ============================================================

def test_unauthorized_endpoints(client):
    # Attempting to call endpoints without X-API-Key header should result in 401/403 error
    endpoints = [
        ("/analyze-sentiment", "post", {"text": "hello"}),
        ("/chatbot", "post", {"message": "hello"}),
        ("/dashboard/stats", "get", None)
    ]
    for endpoint, method, payload in endpoints:
        if method == "post":
            response = client.post(endpoint, json=payload)
        else:
            response = client.get(endpoint)
        assert response.status_code in (401, 403)


# ============================================================
# Sentiment Analysis
# ============================================================

def test_analyze_sentiment(client):
    payload = {"text": "I absolutely love this dress! The fabric is high quality and it fits perfectly."}
    response = client.post("/analyze-sentiment", json=payload, headers=API_KEY_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Positive"
    assert data["confidence"] > 0.0

    payload_neg = {"text": "Avoid this! The customer service was rude when I requested a refund."}
    response = client.post("/analyze-sentiment", json=payload_neg, headers=API_KEY_HEADERS)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "Negative"


# ============================================================
# Chatbot FAQ Queries
# ============================================================

def test_chatbot_faq(client):
    # Test rule-based match
    payload = {"message": "What are your hours?"}
    response = client.post("/chatbot", json=payload, headers=API_KEY_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "store_hours"
    assert data["method"] == "rule"
    assert any(k in data["reply"].lower() for k in ["timings", "open", "hours"])

    # Test ML classification match
    payload_ml = {"message": "Can I return a purchase?"}
    response = client.post("/chatbot", json=payload_ml, headers=API_KEY_HEADERS)
    assert response.status_code == 200
    data_ml = response.json()
    assert data_ml["intent"] == "return_policy"


# ============================================================
# Product Classification
# ============================================================

def test_product_classification(client):
    # Generate a dummy image in memory representing a product image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    
    file_bytes = io.BytesIO(buffer.tobytes())
    response = client.post(
        "/classify-product",
        files={"file": ("product.jpg", file_bytes, "image/jpeg")},
        headers=API_KEY_HEADERS
    )
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert "probabilities" in data


# ============================================================
# Face Recognition & Visited Logging
# ============================================================

def test_recognize_returning_customer(client):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    
    file_bytes = io.BytesIO(buffer.tobytes())
    response = client.post(
        "/recognize-face",
        files={"file": ("customer.jpg", file_bytes, "image/jpeg")},
        headers=API_KEY_HEADERS
    )
    assert response.status_code == 200
    data = response.json()
    assert "face_detected" in data


# ============================================================
# Dashboard Statistics
# ============================================================

def test_dashboard_stats(client):
    response = client.get("/dashboard/stats", headers=API_KEY_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "total_visits" in data
    assert "recognized_customer_visits" in data
    assert "anonymous_customer_visits" in data
    assert "sentiment_distribution" in data
    assert "recent_visits" in data