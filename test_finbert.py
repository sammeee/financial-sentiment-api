from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Use the model's own label mapping
id2label = model.config.id2label
print("Model's label mapping:", id2label)

text = "Apple reported record quarterly earnings."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()[0]

# Get predicted class and label
predicted_class_id = probs.argmax()
sentiment = id2label[predicted_class_id]
confidence = probs[predicted_class_id]

print(f"Predicted: {sentiment} (confidence: {confidence:.4f})")
print(f"All probabilities: { {id2label[i]: float(probs[i]) for i in range(len(probs))} }")