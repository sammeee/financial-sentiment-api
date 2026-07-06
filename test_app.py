# test_app.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from Hugging Face Spaces!"}

@app.post("/predict")
def predict():
    return {"sentiment": "test ok"}