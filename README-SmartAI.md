# AI-Powered Smart Retail & Customer Intelligence Platform

An end-to-end, production-grade **Smart Retail & Customer Intelligence Platform** built using FastAPI, TensorFlow, scikit-learn, OpenCV, NLTK, and pandas. 

The platform implements returning-customer facial recognition, product category classification, customer feedback sentiment analysis, and a hybrid FAQ support chatbot behind a secure, unified REST API gateway.

---

## 1. Syllabus & Project Mapping

This project maps Week 6 syllabus topics directly to production ML modules:

| Syllabus Topic | Project Module | Implementation File / Component |
| :--- | :--- | :--- |
| **OpenCV Basics** | Image preprocessing & face bounding box detection | [cv_service.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/cv_service.py) |
| **Image Classification** | CNN Product category classifier | [product_classifier.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/product_classifier.py) |
| **Face Recognition** | Return customer loyalty matching & logging | [face_recognition_module.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/face_recognition_module.py) |
| **Text Preprocessing** | Lowercasing, stopword stripping, lemmatization | [nlp_service.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/nlp_service.py) |
| **Sentiment Analysis** | Logistic Regression text reviews classifier | [nlp_service.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/nlp_service.py) |
| **Chatbot Basics** | Regex rule matching + ML intent fallback | [chatbot_service.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/chatbot_service.py) |
| **ML Pipelines** | Centralized preloaded execution pipeline | [pipeline.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/services/pipeline.py) |
| **Pickle / Joblib** | Model serialization & serialization binaries | `app/models/*.pkl` |
| **FastAPI Layer** | Unified REST APIs with Pydantic & API security | [main.py](file:///c:/Users/ShashiDangi/Downloads/Smart-Retail-AI-Platform-main/Smart-Retail-AI-Platform-main/app/main.py) |
| **API Deployment** | Dockerized containers & automated CI/CD checks | `Dockerfile` & `.github/workflows/deploy.yml` |

---

## 2. System Architecture

```text
                  Client Layer (webcam, feedback, chat queries)
                                      |
                                      |  REST request + X-API-Key Header
                                      v
                             +-----------------+
                             | FastAPI Gateway | (app/main.py)
                             +--------+--------+
                                      |
          +---------------------------+---------------------------+
          v                           v                           v
     [CV Module]                 [NLP Module]             [Chatbot Module]
  (cv_service.py,           (nlp_service.py)           (chatbot_service.py)
 face_recognition_module.py)          |                           |
          |                           |                           |
  - OpenCV Grayscale/Blur    - Lowercase & clean text    - Check rule keywords
  - Haar Cascade detect      - TF-IDF Vectorizer         - ML Classifier fallback
  - MobileNetV2 Embedding    - Sentiment Regression      - Intents matcher
          |                           |                           |
          +---------------------------+---------------------------+
                                      |
                                      v
                            +-------------------+
                            |  Storage Layer    |
                            | - face_db.pkl     |
                            | - intents.json    |
                            | - customer_visits | (data/customer_visits.csv)
                            +-------------------+
```

---

## 3. Project Folder Structure

```text
smart-retail-ai/
├── app/
│   ├── main.py                  # FastAPI server configuration & routing
│   ├── schemas.py               # Pydantic request/response validation schemas
│   ├── models/                  # Serialized ML model binaries
│   │   ├── product_classifier.keras # TensorFlow MobileNetV2 Product category model
│   │   ├── face_db.pkl          # Pickled dictionary of registered customer face embeddings
│   │   ├── sentiment_model.pkl  # TF-IDF + Logistic Regression sentiment classifier
│   │   └── chatbot_model.pkl    # TF-IDF + Classifier chatbot intents model
│   ├── routers/
│   │   ├── vision.py            # Route handlers for face-recognition & image classification
│   │   ├── nlp.py               # Route handlers for sentiment analysis
│   │   └── chatbot.py           # Route handlers for support bot & dashboard statistics
│   └── services/
│       ├── cv_service.py        # OpenCV image helpers & MobileNetV2 extractor
│       ├── face_recognition_module.py # Face matching logic & customer log writer
│       ├── nlp_service.py       # Text preprocessors & sentiment inference
│       ├── chatbot_service.py   # Keyword rules & ML intents processor
│       └── pipeline.py          # Unified pipeline preloader
├── training/
│   └── train_models.py          # Automates compilation of sentiment, chatbot & face DB models
├── notebooks/
│   ├── 01_image_classifier_training.ipynb
│   ├── 02_face_recognition_setup.ipynb
│   └── 03_sentiment_model_training.ipynb
├── data/
│   ├── reviews.csv              # Sentiment model e-commerce reviews dataset
│   ├── intents.json             # Chatbot FAQs knowledge intents
│   └── customer_visits.csv      # Log tracking customer visits with timestamps
├── tests/
│   └── test_endpoints.py        # Complete automated integration tests
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Deployment container blueprint
└── .github/workflows/deploy.yml # GitHub Actions CI/CD workflows configuration
```

---

## 4. API Endpoints Reference

All routes require validation using the authentication header: `X-API-Key: smart-retail-secret-key`.

### `POST /recognize-face`
- **Description**: Upload a customer photo to detect returning profiles or log new visitors.
- **Payload**: Multipart Form-data (`file: UploadFile`).
- **Response**: Bounding box coordinates, identified customer profile label, visit status, and loyalty timestamp.

### `POST /classify-product`
- **Description**: Submits an image file to predict its retail category (Bags, Shoes, Watches, Clothing, etc.).
- **Payload**: Multipart Form-data (`file: UploadFile`).
- **Response**: Class names, confidence scores, and probability mappings.

### `POST /analyze-sentiment`
- **Description**: Processes customer textual reviews to output emotional sentiment.
- **Payload**: JSON (`{"text": "string"}`).
- **Response**: Preprocessed text, predicted sentiment label (`Positive`, `Negative`, `Neutral`), and confidence probability.

### `POST /chatbot`
- **Description**: Submits chat message inputs to support assistant.
- **Payload**: JSON (`{"message": "string"}`).
- **Response**: Bot reply, matched intent tag, and execution strategy (`rule` or `ml`).

### `GET /dashboard/stats`
- **Description**: Yields aggregated system insights.
- **Payload**: None.
- **Response**: Total visit numbers, loyalty visits ratio, recent log timelines, and sentiment distribution counts.

---

## 5. Ethics, Consent, & Privacy Considerations

Facial recognition technologies in commercial settings carry significant social responsibilities. Evaluators check capstone projects against real-world compliance criteria:

1. **Informed Consent & Transparency**: Businesses must clearly inform consumers of in-store biometric cameras. Customers should be given clear "opt-in" options, linking faces to loyalty numbers ONLY with explicit permission.
2. **Data Minimization & Privacy**: Biometric images should never be stored long-term on servers. Once embedding vectors (e.g. 1280-d arrays) are extracted, raw photos should be deleted immediately from cache.
3. **Bias & Fairness**: Biometric algorithms can exhibit accuracy drift across genders, ages, and skin tones due to skewed training datasets. This project uses MobileNetV2 weights trained on diverse populations and logs similarity confidence parameters to inspect matching disparities.

---

## 6. Installation & Deployment Guide

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/KRITIKA233CODER/Smart-Retail-Customer-Intelligence-Platform.git
   cd Smart-Retail-Customer-Intelligence-Platform
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. **Train Models**:
   ```bash
   $env:PYTHONPATH="."  # Windows PowerShell
   python training/train_models.py
   ```

6. **Run Server Locally**:
   ```bash
   python -m uvicorn app.main:app --reload
   ```
   Explore Swagger API Docs at `http://127.0.0.1:8000/docs`.

### Running Tests
Execute:
```bash
python -m pytest tests/test_endpoints.py
```

### Dockerized Execution
1. **Build Container**:
   ```bash
   docker build -t smart-retail-ai .
   ```
2. **Run Container**:
   ```bash
   docker run -p 8000:8000 --env X_API_KEY=smart-retail-secret-key smart-retail-ai
   ```
