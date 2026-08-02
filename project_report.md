# Capstone Project Report: AI-Powered Smart Retail & Customer Intelligence Platform

---

## 1. Project Title & Metadata
- **Project Title**: AI-Powered Smart Retail & Customer Intelligence Platform
- **Category**: Industrial Capstone Project (AI/ML & MLOps Integration)
- **Frameworks**: FastAPI, TensorFlow/Keras, scikit-learn, OpenCV, NLTK, pandas, NumPy
- **Database Logs**: CSV File Storage (`data/customer_visits.csv`, `data/reviews.csv`)
- **Containerization**: Docker Slim Python 3.11 Runtime
- **CI/CD Pipeline**: GitHub Actions Deployment Workflow

---

## 2. Executive Summary & Business Goal
In modern retail and e-commerce spaces, personalization, customer support, and in-store intelligence determine operational efficiency and customer retention.
This capstone project implements a unified **Smart Retail & Customer Intelligence Platform** designed to solve critical retail challenges:
- **Biometric Loyalty Tracking**: Seamlessly matching return customers at store entrances via OpenCV face detection and MobileNetV2 embedding similarity.
- **Visual Inventory Management**: Instantly classifying uploaded product items into category classes (Bags, Shoes, Watches, Clothing, etc.) to automate indexing.
- **Review Sentiment Analytics**: Preprocessing feedback streams and classifying textual feedback into emotional classes (`Positive`, `Negative`, `Neutral`).
- **Automated Support Gateway**: Resolving frequent consumer inquiries (hours, return policy, payment, orders) using keyword matching and a fallback ML intents classifier.

---

## 3. High-Level System Architecture

```text
======================================================================================
                              CLIENT INTERFACE / POSTMAN
======================================================================================
                                          |
                                          |  HTTPS Requests (including X-API-Key)
                                          v
+------------------------------------------------------------------------------------+
|                                 FASTAPI API GATEWAY                                |
+------------------------------------------------------------------------------------+
                                          |
          +-------------------------------+-------------------------------+
          v                               v                               v
+------------------+             +------------------+            +------------------+
|  Vision Router   |             |    NLP Router    |            |  Chatbot Router  |
+------------------+             +------------------+            +------------------+
  - /recognize-face                - /analyze-sentiment            - /chatbot
  - /classify-product                                              - /dashboard/stats
          |                               |                               |
          v                               v                               v
+------------------+             +------------------+            +------------------+
|   CV Service     |             |   NLP Service    |            | Chatbot Service  |
+------------------+             +------------------+            +------------------+
 - Haar Face Detect               - Preprocessor      - Keyword Rule matcher
 - MobileNetV2 Embedding          - TF-IDF Vectorizer            - ML Intents classifier
          |                               |                               |
          +-------------------------------+-------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|                             PERSISTENT LOGS & STORAGE                              |
| - face_db.pkl (Embeddings)                                                         |
| - intents.json (Bot Knowledge)                                                     |
| - reviews.csv (Sentiment Data)                                                     |
| - customer_visits.csv (Loyalty Timestamp Log)                                      |
+------------------------------------------------------------------------------------+
```

---

## 4. Syllabus Topic Mapping

This project is built around educational requirements mapping directly to Week 6 syllabus parameters:

| Syllabus Area | Application Module | Target Class / Code File |
| :--- | :--- | :--- |
| **OpenCV Basics** | Base grayscaling, blurring, Canny edge detection, face detection. | [cv_service.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/cv_service.py) |
| **Image Classification** | Categories prediction using pre-trained convolutional models. | [product_classifier.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/product_classifier.py) |
| **Face Recognition** | biometrics matching, logging profiles with timestamps. | [face_recognition_module.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/face_recognition_module.py) |
| **Text Preprocessing** | Regex token cleanup, nltk stopword filters, word lemmatization. | [nlp_service.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/nlp_service.py) |
| **Sentiment Analysis** | Pipeline using TF-IDF and Logistic Regression on e-commerce feedback. | [nlp_service.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/nlp_service.py) |
| **Chatbot Basics** | Fallback intents classifier backed by intents rule mapping. | [chatbot_service.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/chatbot_service.py) |
| **ML Pipelines** | Context manager lifecycle pipeline loading all weights at start. | [pipeline.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/pipeline.py) |
| **Pickle / Joblib** | Pickled model storage and similarity dictionary serialization. | `app/models/*.pkl` |
| **FastAPI REST API** | Endpoints with validation checks and API key verification. | [main.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/main.py) |
| **API Deployment** | Lightweight production Docker container builds. | `Dockerfile` |

---

## 5. Module A — Computer Vision Preprocessing
The platform encapsulates image decoding and OpenCV operations in [cv_service.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/cv_service.py):
- **Image Decoding**: Uploaded multipart file buffers are parsed via `np.frombuffer` and converted into BGR images using `cv2.imdecode`.
- **Preprocess Functions**: Includes `grayscale`, `blur_image` (Gaussian blur with customizable kernel size), and `detect_edges` (using OpenCV Canny edge detector).
- **Haar Cascade Detector**: Leverages `cv2.CascadeClassifier` with `haarcascade_frontalface_default.xml` to locate face bounding box coordinates $(x, y, w, h)$ on input images.

---

## 6. Module A — Product Category Classifier
Inventory cataloging is handled by [product_classifier.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/product_classifier.py):
- **Transfer Learning Model**: Employs **MobileNetV2** pre-trained on ImageNet as a feature extractor.
- **Classification Head**: Trained on retail categories (`Bags`, `Bottomwear`, `Shoes`, `Topwear`, `Watches`).
- **Prediction Output**: Outputs top category predictions along with confidence percentage scores and the complete probability mapping across all catalog classes.

---

## 7. Module A — Returning-Customer Face Recognition
Biometric recognition is established in [face_recognition_module.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/face_recognition_module.py):
- **Headless Feature Embeddings**: Avoids dlib C++ compilation dependency constraints by running cropped faces through a headless MobileNetV2 base model configured with global average pooling.
- **Database Matching**: Compares the 1280-dimensional normalized vector against registered customer vectors stored in `face_db.pkl` using cosine similarity (dot product of normalized arrays).
- **Loyalty Logs**: If match similarity exceeds the threshold (e.g. 70%), registers customer identity. If not, auto-generates a unique index (e.g. `cust_a39b22`).
- **Csv Logging**: Appends timestamped visit entries containing timestamp, customer ID, and recognition status to `data/customer_visits.csv`.

---

## 8. Module B — NLP Preprocessing
Text cleaning pipelines reside within [nlp_service.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/nlp_service.py):
- **Dynamic Resource Downloader**: Downloads NLTK corpora (`stopwords`, `wordnet`, `punkt`, `punkt_tab`, `omw-1.4`) automatically at runtime to avoid lookup errors.
- **Cleaning Sequence**:
  1. Standardizes text to lowercase.
  2. Strips punctuation characters.
  3. Tokenizes text into word lists.
  4. Filters out common stop words.
  5. Lemmatizes word tokens to root form (e.g., "running" to "run").

---

## 9. Module B — Sentiment Analysis
Automates customer feedback scoring:
- **Baseline Algorithm**: Custom scikit-learn Pipeline incorporating `TfidfVectorizer` and a `LogisticRegression` classifier.
- **Class Balancing**: Formulates classification boundaries using `class_weight='balanced'` on 3 categorical states: `Positive`, `Negative`, and `Neutral`.
- **Confidence Calibration**: Returns classification probabilities representing prediction confidence metrics.

---

## 10. Module B — Support Chatbot
Implements a rule + ML hybrid support assistant in [chatbot_service.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/chatbot_service.py):
- **Keyword Intent Rules**: Direct substring matches on the query text immediately route matching questions to their mapped responses (e.g., `store_hours`, `return_policy`, `order_status`, `payment_support`).
- **ML Intent Fallback**: For complex syntax, clean queries are routed to a TF-IDF + Logistic Regression intents classifier trained on `intents.json`.
- **Fallback State**: Predictions scoring below the threshold default to the `fallback` tag to prompt clarity.

---

## 11. Module C — Unified Pipeline
Implements the MLOps pipeline structure in [pipeline.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/pipeline.py):
- **Startup Resource Preloading**: Avoids ad-hoc loading latency during REST calls. The pipeline initializes and preloads the product classifier, face recognition databases, and NLP sentiment/chatbot pipelines once at gateway initialization.
- **Global Context State**: Handled globally so endpoints reference the pre-allocated memory models directly.

---

## 12. Module C — REST API Gateway & Security
FastAPI implementation in [main.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/main.py):
- **API Key Header Authentication**: Endpoints require `X-API-Key: smart-retail-secret-key` authorization.
- **Endpoint Structure**:
  - `POST /recognize-face`: Handles loyalty biometrics.
  - `POST /classify-product`: Predicts inventory categories.
  - `POST /analyze-sentiment`: Scores review text.
  - `POST /chatbot`: Returns support responses.
  - `GET /dashboard/stats`: Returns visit metrics and sentiment trends.
- **Interactive Swagger Documentation**: Automatically exposed at `/docs`.

---

## 13. Training & Validation Suite
Model training is automated inside [train_models.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/training/train_models.py):
- **Automated Training Run**: Programmatically prepares training datasets, fits vectorizers, registers biometric face database anchors, and outputs pickles to `app/models/`.
- **Automated Tests**: Formulates full integration testing inside [test_endpoints.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/tests/test_endpoints.py) checking routers, payload schemas, and authentication middleware.

---

## 14. Dockerization & Deployment
Containerization and CI/CD assets:
- **Dockerfile**: Employs Python 3.11-slim base, installs Mesagl/OpenCV bindings, downloads NLTK files during build, copies source code, and mounts the uvicorn gateway.
- **CI/CD Configuration**: GitHub actions runner (`deploy.yml`) checking styling, executing pytest, and packaging container builds.

---

## 15. Ethics, Consent, & Biometric Privacy Report
- **Privacy and Consent**: Retail camera systems must deploy transparent signage. Customer biometric information should be linked only on an opt-in basis.
- **Security & Data Retention**: Avoid storing face images locally. Convert incoming images to embedding arrays instantly and discard raw images.
- **Fairness & Bias Audit**: Periodically inspect matching confidence intervals across demographics to ensure equal performance.

---

## 16. Trade-offs & Future Vision
- **Current Limitations**: Database logs are local CSVs, and chatbot responses are selected from a static intent file.
- **Proposed Upgrades**:
  1. Integrate SQL databases for logging visitor data.
  2. Implement fine-tuned **DistilBERT** transformers for complex sentiment analysis.
  3. Deploy WebSocket endpoints to support real-time camera face recognition.
