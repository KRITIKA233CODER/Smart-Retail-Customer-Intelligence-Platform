from pathlib import Path
import json
import joblib
import tensorflow as tf

class UnifiedPipeline:
    def __init__(self):
        self.models_dir = Path("app/models")
        self.data_dir = Path("data")
        
        self.product_model = None
        self.product_classes = {}
        
        self.face_extractor = None
        self.face_db = {}
        
        self.sentiment_model = None
        
        self.chatbot_model = None
        self.chatbot_intents = {}
        
    def load_all_models(self):
        print("Initializing Unified ML Pipeline and preloading all models...")
        
        # 1. Load Product Image Classifier
        product_model_path = self.models_dir / "product_classifier.keras"
        classes_path = self.models_dir / "product_classes.json"
        
        if product_model_path.exists():
            self.product_model = tf.keras.models.load_model(product_model_path)
            print(" - Loaded product classifier CNN model.")
        if classes_path.exists():
            with open(classes_path, "r", encoding="utf-8") as f:
                self.product_classes = json.load(f)
                
        # 2. Load Face Extractor & Database
        from app.services.cv_service import get_embedding_extractor
        self.face_extractor = get_embedding_extractor()
        print(" - Headless face embedding extractor initialized.")
        
        face_db_path = self.models_dir / "face_db.pkl"
        if face_db_path.exists():
            self.face_db = joblib.load(face_db_path)
            print(f" - Loaded face database with {len(self.face_db)} registered profiles.")
        else:
            self.face_db = {}
            print(" - Warning: face_db.pkl not found, starting with empty database.")
            
        # 3. Load Sentiment Model
        sentiment_model_path = self.models_dir / "sentiment_model.pkl"
        if sentiment_model_path.exists():
            self.sentiment_model = joblib.load(sentiment_model_path)
            print(" - Loaded NLP sentiment model.")
        else:
            print(" - Warning: sentiment_model.pkl not found.")
            
        # 4. Load Chatbot Intents & Classifier
        intents_path = self.data_dir / "intents.json"
        if intents_path.exists():
            with open(intents_path, "r", encoding="utf-8") as f:
                self.chatbot_intents = json.load(f)
        
        chatbot_model_path = self.models_dir / "chatbot_model.pkl"
        if chatbot_model_path.exists():
            self.chatbot_model = joblib.load(chatbot_model_path)
            print(" - Loaded support chatbot intent classifier.")
        else:
            print(" - Warning: chatbot_model.pkl not found.")
            
        print("Unified ML Pipeline load completed successfully.")

pipeline = UnifiedPipeline()
