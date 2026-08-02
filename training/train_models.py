import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import cv2

MODELS_DIR = Path("app/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_sentiment_model():
    print("Training Sentiment Analysis Model...")
    reviews_file = Path("data/reviews.csv")
    if not reviews_file.exists():
        raise FileNotFoundError("data/reviews.csv is missing!")
        
    df = pd.read_csv(reviews_file)
    
    from app.services.nlp_service import clean_text
    df["CleanText"] = df["Text"].apply(clean_text)
    
    X = df["CleanText"]
    y = df["Sentiment"]
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    
    pipeline.fit(X, y)
    
    joblib.dump(pipeline, MODELS_DIR / "sentiment_model.pkl")
    print("Sentiment analysis model saved to app/models/sentiment_model.pkl")

def train_chatbot_model():
    print("Training Chatbot Intent Model...")
    intents_file = Path("data/intents.json")
    if not intents_file.exists():
        raise FileNotFoundError("data/intents.json is missing!")
        
    with open(intents_file, "r", encoding="utf-8") as f:
        intents_data = json.load(f)
        
    patterns = []
    tags = []
    
    from app.services.nlp_service import clean_text
    
    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            patterns.append(clean_text(pattern))
            tags.append(intent["tag"])
            
    if not patterns:
        raise ValueError("No patterns found in intents.json!")
        
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    
    pipeline.fit(patterns, tags)
    
    joblib.dump(pipeline, MODELS_DIR / "chatbot_model.pkl")
    print("Chatbot model saved to app/models/chatbot_model.pkl")

def setup_face_db():
    print("Setting up Face Recognition Database...")
    from app.services.cv_service import get_embedding_extractor, extract_face_embedding
    extractor = get_embedding_extractor()
    
    alice_face = np.zeros((100, 100, 3), dtype=np.uint8)
    alice_face[:, :] = [255, 0, 0] # BGR Blue
    
    bob_face = np.zeros((100, 100, 3), dtype=np.uint8)
    bob_face[:, :] = [0, 255, 0] # BGR Green
    
    alice_emb = extract_face_embedding(alice_face, extractor)
    bob_emb = extract_face_embedding(bob_face, extractor)
    
    face_db = {
        "customer_alice": alice_emb,
        "customer_bob": bob_emb
    }
    
    joblib.dump(face_db, MODELS_DIR / "face_db.pkl")
    print(f"Face database saved with profiles: {list(face_db.keys())}")

if __name__ == "__main__":
    train_sentiment_model()
    train_chatbot_model()
    setup_face_db()
    print("Model training setup complete.")
