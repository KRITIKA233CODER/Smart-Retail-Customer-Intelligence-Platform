import json
import random
from app.services.nlp_service import clean_text

def load_intents(intents_path="data/intents.json"):
    with open(intents_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_rule_based_intent(message: str) -> str:
    msg = message.lower()
    
    # Check hours rules
    if any(k in msg for k in ["hour", "timings", "open", "close", "sunday"]):
        return "store_hours"
    # Check return rules
    if any(k in msg for k in ["return", "refund", "exchange"]):
        return "return_policy"
    # Check order tracking rules
    if any(k in msg for k in ["order", "track", "package", "delivery", "ship"]):
        return "order_status"
    # Check payment support rules
    if any(k in msg for k in ["pay", "payment", "card", "credit", "paypal", "apple pay", "google pay"]):
        return "payment_support"
    # Check greeting rules
    if any(k in msg for k in ["hi", "hello", "hey", "good morning", "good day"]):
        return "greeting"
    # Check goodbye rules
    if any(k in msg for k in ["bye", "goodbye", "see you"]):
        return "goodbye"
        
    return None

def get_chatbot_reply(message: str, intents_data, chatbot_model, threshold=0.40) -> dict:
    # 1. Try rule-based matching
    intent_tag = get_rule_based_intent(message)
    method = "rule"
    
    # 2. Try ML fallback if no rule matched
    if not intent_tag:
        method = "ml"
        cleaned_msg = clean_text(message)
        if cleaned_msg.strip() == "":
            intent_tag = "fallback"
        else:
            probs = chatbot_model.predict_proba([cleaned_msg])[0]
            classes = chatbot_model.classes_
            prob_dict = dict(zip(classes, probs))
            predicted_tag = chatbot_model.predict([cleaned_msg])[0]
            confidence = float(prob_dict.get(predicted_tag, 0.0))
            
            if confidence >= threshold:
                intent_tag = predicted_tag
            else:
                intent_tag = "fallback"
                
    # 3. Retrieve response from intents.json
    intents_list = intents_data.get("intents", [])
    matched_intent = next((item for item in intents_list if item["tag"] == intent_tag), None)
    
    if matched_intent and matched_intent.get("responses"):
        reply = random.choice(matched_intent["responses"])
    else:
        reply = "I'm sorry, I didn't quite understand that. Can you please rephrase your query?"
        
    return {
        "message": message,
        "reply": reply,
        "intent": intent_tag,
        "method": method
    }
