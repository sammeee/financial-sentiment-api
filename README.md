# 📈 Financial Sentiment API

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![FinBERT](https://img.shields.io/badge/Model-FinBERT-orange)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue)

A production-ready REST API for financial news sentiment analysis powered by FinBERT.

---

## 🚀 Live API

**Base URL → [https://szamadesh-financial-sentiment-api.hf.space](https://szamadesh-financial-sentiment-api.hf.space)**

**Swagger Docs → [https://szamadesh-financial-sentiment-api.hf.space/docs](https://szamadesh-financial-sentiment-api.hf.space/docs)**

---

## ✨ Key Features

- 🎯 Sentiment classification — Positive, Negative, or Neutral
- 📊 Confidence scores and full probability distribution
- ⚡ Sub-second inference with FinBERT (ProsusAI/finbert)
- 🐳 Dockerised and deployed on Hugging Face Spaces
- 📝 Auto-generated Swagger docs at `/docs`

---

## 🛠️ Technologies Used

- **FinBERT** — BERT model fine-tuned on financial text
- **FastAPI** — REST API with Pydantic validation
- **PyTorch** — model inference
- **Docker** — containerisation
- **Hugging Face Spaces** — deployment

---

## 📋 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict sentiment |

---

## 💻 Usage

Send a POST request to `/predict`:

```json
{
  "text": "Apple reported record quarterly earnings."
}
```

Example response:

```json
{
  "text": "Apple reported record quarterly earnings.",
  "sentiment": "positive",
  "confidence": 0.9823,
  "probabilities": {
    "positive": 0.9823,
    "neutral": 0.0134,
    "negative": 0.0043
  }
}
```

---

## 🗂️ Project Structure

```text
financial-sentiment-api/
├── api/
│   └── main.py          # FastAPI app with FinBERT inference
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/sammeee/financial-sentiment-api.git
cd financial-sentiment-api
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 🐳 Docker

```bash
docker build -t financial-sentiment-api .
docker run -p 7860:7860 financial-sentiment-api
```

---

## 📊 Results & Observations

- FinBERT correctly classifies financial headlines with high confidence
- Positive news (earnings beats, profit growth) scores above 0.95 confidence
- Negative news (losses, defaults, regulatory fines) scores above 0.90 confidence
- Neutral headlines (routine announcements) correctly avoided as positive or negative

---

## 🔮 Future Improvements

- [ ] Batch prediction endpoint for multiple headlines
- [ ] Rate limiting and API key authentication
- [ ] Fine-tune on more recent financial news data
- [ ] Add a Streamlit frontend for non-technical users