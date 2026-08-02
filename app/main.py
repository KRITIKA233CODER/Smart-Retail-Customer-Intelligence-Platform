from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers.vision import router as vision_router
from app.routers.nlp import router as nlp_router
from app.routers.chatbot import router as chatbot_router
from app.services.pipeline import pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models once at startup
    pipeline.load_all_models()
    yield
    print("Shutting down Smart Retail AI Platform...")

app = FastAPI(
    title="Smart Retail AI Platform",
    description="AI-powered smart retail and customer intelligence platform.",
    version="1.0.0",
    lifespan=lifespan
)

# Register routers
app.include_router(vision_router)
app.include_router(nlp_router)
app.include_router(chatbot_router)

@app.get("/")
def root():
    return {
        "message": "Smart Retail & Customer Intelligence Platform API",
        "status": "running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }