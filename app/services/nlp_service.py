import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib

_nltk_downloaded = False

def download_nltk_resources():
    global _nltk_downloaded
    if not _nltk_downloaded:
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('tokenizers/punkt')
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                # If punkt_tab lookup fails, trigger download in except block
                raise LookupError()
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        _nltk_downloaded = True

def clean_text(text: str) -> str:
    download_nltk_resources()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    ]
    return " ".join(cleaned)

def analyze_sentiment_text(text: str, sentiment_model) -> dict:
    """Predict text sentiment (Positive/Negative/Neutral) using the preloaded pipeline."""
    cleaned = clean_text(text)
    # The sentiment_model is a Pipeline containing TF-IDF and LogisticRegression/SVM
    prediction = sentiment_model.predict([cleaned])[0]
    probs = sentiment_model.predict_proba([cleaned])[0]
    classes = sentiment_model.classes_
    prob_dict = dict(zip(classes, probs))
    confidence = float(prob_dict.get(prediction, 0.0))
    
    return {
        "text": text,
        "sentiment": prediction,
        "confidence": round(confidence, 4)
    }
