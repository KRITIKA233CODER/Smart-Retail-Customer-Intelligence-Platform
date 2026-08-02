from pathlib import Path
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from app.schemas import ChatbotRequest, ChatbotResponse, DashboardStatsResponse
from app.services.chatbot_service import get_chatbot_reply
from app.services.pipeline import pipeline
from app.routers.nlp import verify_api_key

router = APIRouter(
    tags=["Chatbot & Analytics"]
)

@router.post("/chatbot", response_model=ChatbotResponse)
def chatbot(request: ChatbotRequest, authenticated: bool = Depends(verify_api_key)):
    if not pipeline.chatbot_model:
        raise HTTPException(status_code=500, detail="Chatbot model is not initialized.")
    try:
        reply = get_chatbot_reply(request.message, pipeline.chatbot_intents, pipeline.chatbot_model)
        return reply
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats(authenticated: bool = Depends(verify_api_key)):
    visits_file = Path("data/customer_visits.csv")
    total_visits = 0
    recognized_visits = 0
    new_visits = 0
    recent_visits_list = []
    
    if visits_file.exists():
        try:
            df_visits = pd.read_csv(visits_file)
            total_visits = len(df_visits)
            recognized_visits = int((df_visits["status"] == "Recognized").sum())
            new_visits = int((df_visits["status"] == "New").sum())
            
            # Get last 5 visits
            recent_df = df_visits.tail(5).iloc[::-1]
            for _, row in recent_df.iterrows():
                recent_visits_list.append({
                    "timestamp": str(row["timestamp"]),
                    "customer_id": str(row["customer_id"]),
                    "status": str(row["status"])
                })
        except Exception as e:
            print(f"Error reading customer visits log: {e}")
            
    reviews_file = Path("data/reviews.csv")
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    if reviews_file.exists():
        try:
            df_reviews = pd.read_csv(reviews_file)
            counts = df_reviews["Sentiment"].value_counts().to_dict()
            for key in sentiment_counts:
                sentiment_counts[key] = int(counts.get(key, 0))
        except Exception as e:
            print(f"Error reading reviews dataset: {e}")
            
    return {
        "total_visits": total_visits,
        "recognized_customer_visits": recognized_visits,
        "anonymous_customer_visits": new_visits,
        "recent_visits": recent_visits_list,
        "sentiment_distribution": sentiment_counts
    }
