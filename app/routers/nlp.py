from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from app.schemas import SentimentRequest, SentimentResponse
from app.services.nlp_service import analyze_sentiment_text
from app.services.pipeline import pipeline

router = APIRouter(
    tags=["Natural Language Processing"]
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != "smart-retail-secret-key":
        raise HTTPException(
            status_code=403,
            detail="Unauthorized: Invalid API Key"
        )

@router.post("/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest, authenticated: bool = Depends(verify_api_key)):
    if not pipeline.sentiment_model:
        raise HTTPException(status_code=500, detail="Sentiment model is not initialized.")
    try:
        result = analyze_sentiment_text(request.text, pipeline.sentiment_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
