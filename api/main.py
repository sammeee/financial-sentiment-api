# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Financial Sentiment API", 
              description="Predict sentiment of financial news headlines using FinBERT",
              version="1.0.0")

# Load model and tokenizer once at startup
model_name = "ProsusAI/finbert"
logger.info(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
logger.info(f"Model loaded on {device}")

# Define request and response schemas
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str  # "positive", "neutral", "negative"
    confidence: float
    probabilities: dict  # e.g., {"positive": 0.95, "neutral": 0.03, "negative": 0.02}

# Map label IDs to sentiment names
id2label = model.config.id2label

@app.get("/")
def root():
    return {"message": "Financial Sentiment Analysis API is running. Use POST /predict"}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    """
    Predict sentiment of a financial news headline.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Get predicted class
    predicted_class_id = probabilities.argmax()
    sentiment = id2label[predicted_class_id]
    confidence = float(probabilities[predicted_class_id])
    
    # Format probabilities as dict
    prob_dict = {id2label[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    return SentimentResponse(
        text=text,
        sentiment=sentiment,
        confidence=confidence,
        probabilities=prob_dict
    )