# Smart Retail AI Platform

An end-to-end **AI and Machine Learning platform for retail intelligence**, built using Python, FastAPI, TensorFlow, scikit-learn, OpenCV, pandas, and NumPy.

The project integrates multiple machine learning capabilities into a single modular backend API:

- Product Image Classification
- Computer Vision Preprocessing
- Customer Segmentation
- Recommendation System
- Sales Forecasting
- Fraud Detection

The objective of the project is to demonstrate how different machine learning techniques can be trained, evaluated, persisted, and integrated into a unified retail analytics platform.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Machine Learning Modules](#machine-learning-modules)
5. [Technology Stack](#technology-stack)
6. [Project Structure](#project-structure)
7. [API Endpoints](#api-endpoints)
8. [Installation](#installation)
9. [Running the Application](#running-the-application)
10. [API Examples](#api-examples)
11. [Datasets](#datasets)
12. [Model Training](#model-training)
13. [Results](#results)
14. [Testing](#testing)
15. [Design Decisions](#design-decisions)
16. [Limitations](#limitations)
17. [Future Improvements](#future-improvements)
18. [Disclaimer](#disclaimer)

---

# Overview

Modern retail systems generate data across products, customers, transactions, sales, and payments.

The **Smart Retail AI Platform** combines several machine learning techniques to demonstrate how AI can be applied across these areas.

The platform contains five primary ML modules:

| Module | Problem |
|---|---|
| Product Classification | Identify product category from an image |
| Customer Segmentation | Group customers by behavioral characteristics |
| Recommendation System | Recommend items using historical ratings |
| Sales Forecasting | Predict future daily sales |
| Fraud Detection | Detect potentially fraudulent transactions |

All trained models are exposed through a **FastAPI REST API**.

The project separates:

```text
Data
  |
  v
Preprocessing
  |
  v
Training
  |
  v
Evaluation
  |
  v
Saved Model Artifacts
  |
  v
Service Layer
  |
  v
FastAPI Routers
  |
  v
REST API
```

This separation keeps model development independent from inference and API logic.

---

# Features

## 1. Product Image Classification

The product classification module identifies retail products from images.

Supported classes:

- Bags
- Bottomwear
- Shoes
- Topwear
- Watches

The model uses **MobileNetV2 transfer learning**.

Instead of training a convolutional neural network from scratch, a pretrained MobileNetV2 network is used as the visual feature extractor with a classification head trained for the five selected product categories.

The final model is stored as:

```text
app/models/product_classifier.keras
```

The class mapping is stored as:

```text
app/models/product_classes.json
```

---

## 2. Computer Vision Utilities

The platform also exposes OpenCV-based image processing functionality.

Supported operations include:

- Image decoding
- Grayscale conversion
- Image blurring
- Edge detection
- Face detection

These operations are exposed through the `/vision` API routes.

---

## 3. Customer Segmentation

The customer segmentation module groups customers according to:

- Age
- Annual Income
- Spending Score

The system uses **K-Means clustering**.

Before clustering, the features are standardized using `StandardScaler`.

Multiple values of K were evaluated using the silhouette score.

The best result among the tested configurations was:

```text
K = 6
Silhouette Score = 0.4284
```

Cluster sizes:

```text
Cluster 0: 45 customers
Cluster 1: 39 customers
Cluster 2: 33 customers
Cluster 3: 39 customers
Cluster 4: 23 customers
Cluster 5: 21 customers
```

Saved artifacts:

```text
app/models/customer_kmeans.pkl
app/models/customer_scaler.pkl
```

---

## 4. Recommendation System

The recommendation module provides personalized recommendations using **Collaborative Filtering**.

Implemented strategies:

- User-based Collaborative Filtering
- Item-based Collaborative Filtering
- Popularity-based Recommendations
- Cold-start fallback

The system creates a user-item rating matrix and calculates similarity between users and items.

Saved artifacts include:

```text
app/models/user_item_matrix.pkl
app/models/user_similarity.pkl
app/models/item_similarity.pkl
app/models/popular_items.pkl
app/models/recommendation_items.pkl
```

For users without historical interactions, the system automatically falls back to popularity-based recommendations.

---

## 5. Sales Forecasting

The sales forecasting module predicts future daily retail sales from historical transaction data.

The pipeline includes:

- Transaction cleaning
- Cancelled-order removal
- Invalid quantity removal
- Invalid price removal
- Revenue calculation
- Daily sales aggregation
- Calendar features
- Lag features
- Rolling statistics
- Chronological train/test splitting
- Recursive future forecasting

Two regression models were compared:

- Linear Regression
- Random Forest Regressor

Random Forest achieved the better evaluation result and was selected as the final model.

Saved artifacts:

```text
app/models/sales_forecast_model.pkl
app/models/sales_forecast_features.pkl
```

---

## 6. Fraud Detection

The fraud detection module classifies credit-card transactions as either:

```text
Legitimate
```

or:

```text
Fraud
```

The dataset is extremely imbalanced, with fraud representing approximately **0.17%** of transactions.

Two models were evaluated:

- Logistic Regression
- Random Forest Classifier

Random Forest produced substantially better precision and F1 performance and was selected as the final model.

Final model:

```text
app/models/fraud_detection_model.pkl
```

---

# System Architecture

```text
                         Client
                           |
                           v
                    +--------------+
                    |   FastAPI    |
                    | REST Backend |
                    +------+-------+
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
+---------------+   +---------------+   +---------------+
|    Computer   |   |    Customer   |   |     Retail    |
|     Vision    |   |   Analytics   |   |   Analytics   |
+-------+-------+   +-------+-------+   +-------+-------+
        |                   |                   |
        v                   v                   |
  MobileNetV2            K-Means                |
      OpenCV                                    |
                                                |
                          +---------------------+-------------------+
                          |                     |                   |
                          v                     v                   v
                 +----------------+    +---------------+    +---------------+
                 | Recommendation |    |     Sales     |    |     Fraud     |
                 |     System     |    |  Forecasting  |    |   Detection   |
                 +-------+--------+    +-------+-------+    +-------+-------+
                         |                     |                    |
                         v                     v                    v
                 Collaborative          Random Forest        Random Forest
                   Filtering              Regressor            Classifier
```

---

# Machine Learning Modules

| Module | ML Problem | Algorithm / Technique |
|---|---|---|
| Product Classification | Image Classification | MobileNetV2 Transfer Learning |
| Customer Segmentation | Unsupervised Learning | K-Means |
| Recommendation System | Recommender System | Collaborative Filtering |
| Sales Forecasting | Regression / Time Series | Random Forest Regressor |
| Fraud Detection | Imbalanced Classification | Random Forest Classifier |

---

# Technology Stack

## Backend

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic

## Machine Learning

- TensorFlow
- Keras
- scikit-learn
- NumPy
- pandas
- joblib

## Computer Vision

- OpenCV
- MobileNetV2

## Data Processing

- pandas
- NumPy
- openpyxl

## Visualization

- Matplotlib

## Dataset Management

- KaggleHub

## Testing

- pytest
- FastAPI TestClient
- httpx2

## Version Control

- Git
- GitHub

---

# Project Structure

```text
smart-retail-ai/
|
+-- app/
|   |
|   +-- __init__.py
|   +-- main.py
|   |
|   +-- models/
|   |   +-- customer_kmeans.pkl
|   |   +-- customer_scaler.pkl
|   |   +-- fraud_detection_model.pkl
|   |   +-- item_similarity.pkl
|   |   +-- popular_items.pkl
|   |   +-- product_classes.json
|   |   +-- product_classifier.keras
|   |   +-- recommendation_items.pkl
|   |   +-- sales_forecast_features.pkl
|   |   +-- sales_forecast_model.pkl
|   |   +-- user_item_matrix.pkl
|   |   +-- user_similarity.pkl
|   |
|   +-- routers/
|   |   +-- __init__.py
|   |   +-- vision.py
|   |   +-- segmentation.py
|   |   +-- recommendation.py
|   |   +-- forecast.py
|   |   +-- fraud.py
|   |
|   +-- services/
|       +-- __init__.py
|       +-- cv_service.py
|       +-- product_classifier.py
|       +-- segmentation_service.py
|       +-- recommendation_service.py
|       +-- forecast_service.py
|       +-- fraud_service.py
|
+-- training/
|   |
|   +-- download_dataset.py
|   +-- inspect_dataset.py
|   +-- prepare_product_dataset.py
|   +-- train_product_classifier.py
|   |
|   +-- inspect_customers.py
|   +-- train_customer_segmentation.py
|   |
|   +-- inspect_recommendations.py
|   +-- train_recommender.py
|   +-- evaluate_recommender.py
|   +-- test_recommender.py
|   |
|   +-- inspect_sales.py
|   +-- prepare_sales_data.py
|   +-- train_sales_forecast.py
|   +-- test_forecast.py
|   |
|   +-- inspect_fraud.py
|   +-- train_fraud_detection.py
|   +-- test_fraud.py
|
+-- tests/
|   +-- test_endpoints.py
|
+-- data/
|   |
|   +-- customers/
|   |   +-- Mall_Customers.csv
|   |   +-- results/
|   |
|   +-- fashion/
|   |   +-- styles.csv
|   |
|   +-- fraud/
|   |   +-- results/
|   |
|   +-- recommendations/
|   |   +-- ml-100k/
|   |
|   +-- sales/
|       +-- processed/
|       +-- results/
|
+-- .gitignore
+-- requirements.txt
+-- README.md
```

Large raw datasets and generated image collections are intentionally excluded from version control.

---

# API Endpoints

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/vision/classify-product` | Product image classification |
| POST | `/vision/preprocess` | Image preprocessing |
| POST | `/vision/detect-face` | Face detection |
| POST | `/segment-customer` | Customer segmentation |
| POST | `/recommend` | Personalized recommendations |
| POST | `/forecast` | Future sales forecasting |
| POST | `/fraud-check` | Fraud prediction |

FastAPI automatically generates interactive API documentation.

After starting the application, Swagger UI is available at:

```text
http://127.0.0.1:8000/docs
```

OpenAPI schema:

```text
http://127.0.0.1:8000/openapi.json
```

---

# Installation

## Prerequisites

Recommended environment:

```text
Python 3.11
```

Git is also required if cloning the repository.

---

## 1. Clone the Repository

```bash
git clone https://github.com/DhruvChoudhary1/Smart-Retail-AI-Platform.git
cd smart-retail-ai
```

---

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then:

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

# Running the Application

Start the FastAPI server from the project root:

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

Expected health response:

```json
{
  "status": "healthy"
}
```

The root endpoint returns:

```json
{
  "message": "Smart Retail AI Platform API",
  "status": "running"
}
```

---

# API Examples

## Product Classification

Endpoint:

```text
POST /vision/classify-product
```

The endpoint accepts an uploaded image file using multipart form data.

Example request using curl:

```bash
curl -X POST \
  -F "file=@product.jpg" \
  http://127.0.0.1:8000/vision/classify-product
```

The service:

1. Reads the uploaded image.
2. Decodes the image using OpenCV.
3. Preprocesses the image for MobileNetV2.
4. Runs model inference.
5. Returns the predicted category and model output.

---

## Image Preprocessing

Endpoint:

```text
POST /vision/preprocess
```

The endpoint performs:

- Grayscale conversion
- Blurring
- Edge detection

The processed images are returned as Base64-encoded JPEG data.

---

## Face Detection

Endpoint:

```text
POST /vision/detect-face
```

The endpoint detects faces in an uploaded image and returns the detected face locations.

---

## Customer Segmentation

Endpoint:

```text
POST /segment-customer
```

Example request:

```json
{
  "age": 28,
  "annual_income": 70,
  "spending_score": 85
}
```

`annual_income` represents annual income in thousands.

The service:

1. Validates the request.
2. Standardizes the input features.
3. Runs K-Means prediction.
4. Returns the assigned customer cluster.

---

## Recommendation System

Endpoint:

```text
POST /recommend
```

Example:

```json
{
  "user_id": 196,
  "k": 5,
  "method": "item"
}
```

Supported recommendation methods:

```text
item
user
popular
```

### Cold Start

If the user is unknown:

```json
{
  "user_id": 9999,
  "k": 5,
  "method": "item"
}
```

the recommendation service automatically falls back to:

```text
popular
```

recommendations.

---

## Sales Forecast

Endpoint:

```text
POST /forecast
```

Example:

```json
{
  "days": 7
}
```

The API supports forecasts from 1 to 30 future days.

The service recursively generates future sales predictions using the historical observations and saved Random Forest model.

---

## Fraud Detection

Endpoint:

```text
POST /fraud-check
```

The endpoint expects the same feature representation as the Credit Card Fraud Detection dataset:

```text
Time
V1
V2
...
V28
Amount
```

Example structure:

```json
{
  "Time": 0,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

The response contains fields including:

```text
prediction
is_fraud
classification
fraud_probability
fraud_probability_percent
legitimate_probability
```

---

# Datasets

The project uses multiple public datasets for the different machine learning modules.

Large source datasets are not committed to Git.

---

## 1. Fashion Product Images

Used for:

```text
Product Classification
```

Dataset:

```text
Fashion Product Images Small
```

Kaggle identifier:

```text
paramaggarwal/fashion-product-images-small
```

The original dataset contains approximately 44,000 product images and metadata.

The project selects five categories:

```text
Topwear
Shoes
Bags
Bottomwear
Watches
```

Prepared dataset:

```text
Train:      11,387 images
Validation:  2,848 images
Total:      14,235 images
```

The generated image directories are excluded from Git.

The source dataset can be downloaded using:

```bash
python training/download_dataset.py
```

---

## 2. Mall Customers

Used for:

```text
Customer Segmentation
```

Dataset size:

```text
200 customers
```

Features:

```text
CustomerID
Genre
Age
Annual Income (k$)
Spending Score (1-100)
```

Features used by the clustering model:

```text
Age
Annual Income (k$)
Spending Score (1-100)
```

---

## 3. MovieLens 100K

Used for:

```text
Recommendation System
```

Dataset statistics:

```text
Users:        943
Items:      1,682
Ratings:  100,000
```

Average rating:

```text
3.53
```

User-item matrix density:

```text
6.30%
```

Sparsity:

```text
93.70%
```

---

## 4. Online Retail

Used for:

```text
Sales Forecasting
```

Original dataset size:

```text
541,909 rows
```

Columns:

```text
InvoiceNo
StockCode
Description
Quantity
InvoiceDate
UnitPrice
CustomerID
Country
```

Date range:

```text
2010-12-01 to 2011-12-09
```

The preprocessing pipeline removes:

- Cancelled transactions
- Quantity <= 0
- UnitPrice <= 0

The cleaned transaction data is aggregated into daily sales.

The raw Excel dataset is excluded from Git because of its size.

---

## 5. Credit Card Fraud Detection

Used for:

```text
Fraud Detection
```

Dataset statistics:

```text
Total transactions:       284,807
Legitimate transactions:  284,315
Fraud transactions:           492
Fraud percentage:           0.1727%
```

Features:

```text
Time
V1 - V28
Amount
Class
```

The raw dataset is excluded from Git because of its size.

---

# Model Training

Training and preprocessing scripts are stored under:

```text
training/
```

The general pipeline is:

```text
Raw Dataset
     |
     v
Dataset Inspection
     |
     v
Cleaning / Preparation
     |
     v
Feature Engineering
     |
     v
Model Training
     |
     v
Evaluation
     |
     v
Model Serialization
     |
     v
FastAPI Integration
```

---

# Product Classification Training

Download the dataset:

```bash
python training/download_dataset.py
```

Inspect metadata:

```bash
python training/inspect_dataset.py
```

Prepare selected product categories:

```bash
python training/prepare_product_dataset.py
```

Train the classifier:

```bash
python training/train_product_classifier.py
```

The model uses:

```text
MobileNetV2
+
Global Average Pooling
+
Dropout
+
Dense Classification Layer
```

The pretrained MobileNetV2 feature extractor significantly reduces the number of parameters that must be trained.

---

# Customer Segmentation Training

Inspect the dataset:

```bash
python training/inspect_customers.py
```

Train:

```bash
python training/train_customer_segmentation.py
```

K values from 2 through 10 were evaluated.

Silhouette results:

```text
K=2   0.3355
K=3   0.3578
K=4   0.4040
K=5   0.4166
K=6   0.4284
K=7   0.4172
K=8   0.4082
K=9   0.4177
K=10  0.4066
```

Best configuration:

```text
K = 6
Silhouette Score = 0.4284
```

---

# Recommendation System Training

Inspect:

```bash
python training/inspect_recommendations.py
```

Train:

```bash
python training/train_recommender.py
```

Evaluate:

```bash
python training/evaluate_recommender.py
```

The training process creates:

- User-item matrix
- User similarity matrix
- Item similarity matrix
- Popularity ranking
- Item metadata

The recommendation service can then use these precomputed artifacts without rebuilding similarity matrices for every API request.

---

# Sales Forecast Training

Inspect:

```bash
python training/inspect_sales.py
```

Prepare:

```bash
python training/prepare_sales_data.py
```

Train:

```bash
python training/train_sales_forecast.py
```

Test:

```bash
python -m training.test_forecast
```

The forecasting dataset contains:

```text
374 daily observations
346 usable feature observations
11 model features
```

Chronological split:

```text
Train observations: 276
Test observations:   70
```

Training period:

```text
2010-12-29 to 2011-09-30
```

Test period:

```text
2011-10-01 to 2011-12-09
```

---

# Fraud Detection Training

Inspect:

```bash
python training/inspect_fraud.py
```

Train:

```bash
python training/train_fraud_detection.py
```

Test:

```bash
python -m training.test_fraud
```

The training/test split contains:

```text
Training transactions: 227,845
Test transactions:      56,962

Training fraud cases: 394
Test fraud cases:      98
```

Because the dataset is extremely imbalanced, model selection is not based on accuracy alone.

Important metrics include:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

# Results

## Customer Segmentation

Best number of clusters:

```text
6
```

Silhouette Score:

```text
0.4284
```

---

## Recommendation System

Item-based Collaborative Filtering evaluation at K=10:

```text
Precision@10: 0.0011
Recall@10:    0.0106
Hit Rate@10:  0.0106
```

The relatively low ranking metrics highlight the limitations of a simple similarity-based recommender on a sparse interaction matrix.

---

## Sales Forecasting

### Linear Regression

```text
MAE:  19,450.01
RMSE: 28,931.59
```

### Random Forest

```text
MAE:  12,788.96
RMSE: 22,999.31
```

Selected model:

```text
Random Forest Regressor
```

---

## Fraud Detection

### Logistic Regression

```text
Precision: 0.0610
Recall:    0.9184
F1 Score:  0.1144
ROC-AUC:   0.9722
```

Confusion matrix:

```text
[[55478  1386]
 [    8    90]]
```

Logistic Regression achieved high recall but generated a large number of false positives.

### Random Forest

```text
Precision: 0.8200
Recall:    0.8367
F1 Score:  0.8283
ROC-AUC:   0.9782
```

Confusion matrix:

```text
[[56846    18]
 [   16    82]]
```

Selected model:

```text
Random Forest Classifier
```

The model detected:

```text
82 of 98 fraud cases
```

while producing:

```text
18 false positives
```

on the test set.

---

# Results Summary

| Module | Final Result |
|---|---|
| Product Classification | MobileNetV2 transfer-learning model for five retail categories |
| Customer Segmentation | K=6, Silhouette Score = 0.4284 |
| Recommendation System | Precision@10 = 0.0011, Recall@10 = 0.0106 |
| Sales Forecasting | RF MAE = 12,788.96, RMSE = 22,999.31 |
| Fraud Detection | Precision = 0.8200, Recall = 0.8367, F1 = 0.8283, ROC-AUC = 0.9782 |

---

# Testing

The project includes automated integration tests for the FastAPI application.

Test file:

```text
tests/test_endpoints.py
```

Run all tests:

```bash
python -m pytest tests -v
```

Latest test result:

```text
13 passed
0 failed
```

The integration test suite validates:

- Root endpoint
- Health endpoint
- Customer segmentation
- Customer input validation
- Recommendation for known users
- Recommendation cold-start handling
- Sales forecasting
- Forecast request validation
- Fraud detection for legitimate transactions
- Fraud detection endpoint behavior
- Fraud request validation
- Product classification
- Vision preprocessing

The final test run completed successfully with:

```text
13 passed, 1 warning
```

The warning is generated by scikit-learn because the saved `StandardScaler` was fitted using named DataFrame features while inference currently supplies equivalent feature values without feature names.

It does not cause a test failure.

---

# Input Validation

FastAPI and Pydantic are used to validate incoming requests.

Examples include:

### Customer Age

```text
0 < age <= 120
```

### Spending Score

```text
1 <= spending_score <= 100
```

### Recommendation Count

```text
1 <= k <= 50
```

### Forecast Horizon

```text
1 <= days <= 30
```

### Fraud Transaction

All required transaction features must be supplied.

Invalid requests automatically receive an HTTP `422 Unprocessable Entity` response when Pydantic validation fails.

---

# Design Decisions

## Transfer Learning for Product Classification

Training a CNN from scratch requires substantial training data and computational resources.

MobileNetV2 provides pretrained visual representations that can be adapted to the five retail categories with a relatively small trainable classification layer.

---

## Standardization Before K-Means

K-Means uses distance calculations.

Without scaling, annual income, age, and spending score could contribute differently simply because of their numeric scales.

`StandardScaler` is therefore applied before clustering.

---

## Silhouette-Based Cluster Selection

K-Means requires the number of clusters to be selected in advance.

Multiple values of K were evaluated, and silhouette score was used as the primary cluster-quality metric.

K=6 produced the highest score among the tested values.

---

## Collaborative Filtering

The recommendation module uses historical user-item interactions rather than manually defined product rules.

Both user-user and item-item similarity matrices are precomputed and stored.

This reduces computation during API requests.

---

## Cold-Start Strategy

Collaborative filtering cannot personalize recommendations for users without historical interactions.

For unknown users, the platform falls back to globally popular items.

---

## Chronological Forecast Validation

Time-series data should not be randomly shuffled before evaluation.

The forecasting module uses earlier observations for training and later observations for testing.

This more closely represents real forecasting behavior and reduces future-data leakage.

---

## Lag and Rolling Features

Historical sales values are transformed into predictive features such as:

- Previous sales
- Historical lags
- Rolling averages

These allow a standard regression model to capture temporal behavior.

---

## Fraud Class Imbalance

Only approximately 0.17% of transactions in the fraud dataset are fraudulent.

A classifier predicting every transaction as legitimate would therefore achieve extremely high accuracy while being useless.

For this reason, the project emphasizes:

```text
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
```

rather than raw accuracy.

---

# Model Artifacts

The trained models are stored under:

```text
app/models/
```

Approximate largest artifact sizes during development:

```text
item_similarity.pkl          ~21.6 MB
user_item_matrix.pkl         ~12.1 MB
product_classifier.keras      ~9.3 MB
user_similarity.pkl           ~6.8 MB
fraud_detection_model.pkl     ~6.8 MB
sales_forecast_model.pkl      ~2.6 MB
```

The trained artifacts are included so that the FastAPI application can perform inference without requiring every model to be retrained first.

---

# Data Version Control

Large datasets are intentionally excluded from Git.

Examples include:

```text
data/products/
data/fraud/creditcard.csv
data/sales/Online Retail.xlsx
```

The Python virtual environment is also excluded:

```text
.venv/
```

Python caches are excluded:

```text
__pycache__/
*.pyc
```

This keeps the repository focused on:

- Source code
- Training pipelines
- Small datasets
- Evaluation results
- Trained model artifacts
- Tests
- Documentation

---

# Limitations

This project is designed as an AI/ML engineering demonstration rather than a production retail platform.

Current limitations include:

### Product Classification

Only five selected product categories are supported.

Real retail catalogs may contain hundreds or thousands of categories.

### Customer Segmentation

The Mall Customers dataset contains only 200 customers and a limited number of behavioral features.

Production segmentation would typically include transaction history, product preferences, visit frequency, lifetime value, and engagement data.

### Recommendation System

The current recommendation engine uses similarity-based collaborative filtering.

Evaluation results show substantial room for improvement.

More advanced systems could use:

- Matrix factorization
- Neural collaborative filtering
- Content-based recommendations
- Hybrid recommendation systems

### Sales Forecasting

The source dataset is historical and ends in December 2011.

The current model also uses recursive forecasting, which can accumulate prediction error over longer forecast horizons.

### Fraud Detection

The fraud dataset contains anonymized PCA-derived features (`V1` to `V28`).

A real fraud system would typically combine:

- Transaction history
- Merchant information
- Device signals
- Geographical signals
- Customer behavior
- Authentication events

The current model should not be interpreted as a production fraud prevention system.

---

# Future Improvements

Potential future extensions include:

## Machine Learning

- XGBoost
- LightGBM
- CatBoost
- Hyperparameter optimization
- Cross-validation
- Threshold optimization
- Probability calibration
- Explainable AI using SHAP

## Product Intelligence

- Product similarity search
- Image embeddings
- Visual search
- Multi-label product classification
- Object detection
- Inventory recognition

## Customer Intelligence

- Customer lifetime value prediction
- Churn prediction
- RFM analysis
- Personalized customer segments
- Behavioral embeddings

## Recommendation Systems

- Matrix factorization
- SVD
- Neural collaborative filtering
- Content-based recommendation
- Hybrid recommendation
- Real-time interaction tracking

## Forecasting

- Gradient boosting
- XGBoost regression
- Temporal cross-validation
- Prophet
- LSTM models
- Transformer-based forecasting
- Inventory demand forecasting

## Fraud Detection

- Decision-threshold optimization
- Cost-sensitive learning
- Gradient boosting
- Anomaly detection
- Real-time fraud scoring
- Explainable fraud decisions

## Platform Engineering

- Docker containerization
- CI/CD pipeline
- Cloud deployment
- Authentication
- Authorization
- API rate limiting
- Centralized logging
- Model monitoring
- Data drift detection
- Model drift detection
- Model versioning

## Frontend

A future web dashboard could provide:

- Product classification interface
- Customer segment visualization
- Recommendation interface
- Forecast charts
- Fraud risk visualization
- Model performance dashboard

---

# Possible Production Architecture

A production version could evolve toward:

```text
                    Web / Mobile Client
                            |
                            v
                      API Gateway
                            |
                            v
                     Authentication
                            |
                            v
                       FastAPI API
                            |
       +--------------------+--------------------+
       |                    |                    |
       v                    v                    v
 Product Service      Customer Service      Risk Service
       |                    |                    |
       v                    v                    v
 Model Serving         Recommendation       Fraud Model
       |                    Engine                |
       +--------------------+--------------------+
                            |
                            v
                       Data Platform
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
      Database          Data Lake         Event Stream
```

This repository focuses on the ML and API layers of such a system.

---

# Reproducibility

The repository contains:

- Dataset inspection scripts
- Data preparation scripts
- Training scripts
- Evaluation scripts
- Saved models
- FastAPI inference services
- Automated endpoint tests

This allows the major development workflow to be inspected and reproduced.

Some large raw datasets are excluded because of repository size constraints and must be downloaded separately.

---

# Development Workflow

A typical workflow for extending the project is:

```text
1. Add / update dataset
          |
          v
2. Inspect dataset
          |
          v
3. Prepare features
          |
          v
4. Train model
          |
          v
5. Evaluate model
          |
          v
6. Save artifact
          |
          v
7. Add service
          |
          v
8. Add API router
          |
          v
9. Add integration test
          |
          v
10. Run complete test suite
```

Before committing major changes:

```bash
python -m pytest tests -v
```

should complete successfully.

---

# Current Project Status

```text
Product Classification       COMPLETE
Computer Vision API          COMPLETE
Customer Segmentation        COMPLETE
Recommendation System        COMPLETE
Sales Forecasting            COMPLETE
Fraud Detection              COMPLETE
FastAPI Integration          COMPLETE
Integration Testing          COMPLETE
Documentation                COMPLETE
```

Current automated integration test result:

```text
13 passed
0 failed
```

---

# Disclaimer

The models in this project were developed using public historical datasets for educational and demonstration purposes.

The platform should not be used directly for:

- Financial decision-making
- Production fraud prevention
- Automated customer eligibility decisions
- High-stakes business decisions

without additional validation, security controls, data governance, monitoring, fairness analysis, domain expertise, and production testing.

---

# License

This project is intended for educational and portfolio use.

Individual datasets may have their own licenses and usage conditions. Refer to the original dataset providers before redistributing or using the datasets commercially.
# Smart-Retail-Customer-Intelligence-Platform
