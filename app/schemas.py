from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# --- Sentiment Schemas ---
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw customer feedback/review text to analyze")

class SentimentResponse(BaseModel):
    text: str
    sentiment: str = Field(..., description="Positive, Negative, or Neutral")
    confidence: float = Field(..., description="Prediction confidence probability score")

# --- Chatbot Schemas ---
class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to the support chatbot")

class ChatbotResponse(BaseModel):
    message: str
    reply: str = Field(..., description="Chatbot response message")
    intent: Optional[str] = Field(None, description="Matched intent tag name")
    method: str = Field(..., description="'rule' or 'ml' fallback matcher")

# --- Vision Schemas ---
class FaceRecognitionResponse(BaseModel):
    filename: str
    face_detected: bool
    customer_id: Optional[str] = None
    status: str = Field(..., description="'Recognized returning customer', 'New customer detected', or 'No face detected'")
    match_score: Optional[float] = None
    timestamp: Optional[str] = None

class ProductClassificationResponse(BaseModel):
    filename: str
    category: str
    confidence: float
    confidence_percent: float
    probabilities: Dict[str, float]

# --- Dashboard Stats Schemas ---
class DashboardStatsResponse(BaseModel):
    total_visits: int
    recognized_customer_visits: int
    anonymous_customer_visits: int
    recent_visits: List[Dict[str, Any]]
    sentiment_distribution: Dict[str, int]
