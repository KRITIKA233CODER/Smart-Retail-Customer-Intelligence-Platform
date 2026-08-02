from pathlib import Path
import datetime
import csv
import numpy as np

VISITS_FILE = Path("data/customer_visits.csv")

def init_visits_log():
    if not VISITS_FILE.exists():
        VISITS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VISITS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "customer_id", "status"])

def log_visit(customer_id, status):
    init_visits_log()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(VISITS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, customer_id, status])
    return timestamp

def recognize_customer_face(image, face_db, extractor_model, threshold=0.70):
    from app.services.cv_service import detect_faces, extract_face_embedding
    
    faces = detect_faces(image)
    if not faces:
        return {
            "face_detected": False,
            "status": "No face detected",
            "customer_id": None,
            "match_score": None,
            "timestamp": None
        }
        
    # Crop the first face
    face_box = faces[0]
    x, y, w, h = face_box["x"], face_box["y"], face_box["width"], face_box["height"]
    cropped_face = image[y:y+h, x:x+w]
    
    # Extract embedding
    embedding = extract_face_embedding(cropped_face, extractor_model)
    embedding_arr = np.array(embedding)
    
    best_match = None
    best_score = -1.0
    
    # Compare with face_db
    if face_db:
        for customer_id, stored_emb in face_db.items():
            stored_emb_arr = np.array(stored_emb)
            similarity = np.dot(embedding_arr, stored_emb_arr)
            if similarity > best_score:
                best_score = similarity
                best_match = customer_id
            
    if best_match and best_score >= threshold:
        status = "Recognized returning customer"
        timestamp = log_visit(best_match, "Recognized")
        return {
            "face_detected": True,
            "status": status,
            "customer_id": best_match,
            "match_score": round(float(best_score), 4),
            "timestamp": timestamp
        }
    else:
        import uuid
        new_id = f"cust_{uuid.uuid4().hex[:6]}"
        status = "New customer detected"
        timestamp = log_visit(new_id, "New")
        return {
            "face_detected": True,
            "status": status,
            "customer_id": new_id,
            "match_score": round(float(best_score), 4) if best_score != -1.0 else 0.0,
            "timestamp": timestamp
        }
