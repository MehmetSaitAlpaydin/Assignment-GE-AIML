import torch
import flask
from flask import Flask, request, jsonify, render_template
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pyngrok import ngrok
import numpy as np

# Load trained model and tokenizer
model_path = "distilbert_newsgroup_model"  # Make sure this is the correct saved model path
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define category labels
categories = [
    "Atheism", "Graphics", "Windows Misc", "IBM Hardware", "Mac Hardware", "Windows X",
    "For Sale", "Autos", "Motorcycles", "Baseball", "Hockey", "Cryptography", "Electronics",
    "Medical", "Space", "Christianity", "Politics - Guns", "Politics - Mideast",
    "Politics - Misc", "Religion - Misc"
]

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "")

        # Tokenize input
        encodings = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        # Get prediction
        model.eval()
        with torch.no_grad():
            logits = model(**encodings).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, top_labels = torch.topk(probs, k=3)

        # Convert to readable format
        predictions = [
            {"category": categories[label.item()], "probability": round(prob.item() * 100, 2)}
            for prob, label in zip(top_probs[0], top_labels[0])
        ]

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)})

# Expose Colab app with ngrok
print("Starting web app...")
ngrok.set_auth_token("1hVkTfUaRO7dgP3Uj2thosOw2HH_5WEaMtenPKmsMQmiJMV7d")  # Replace with your ngrok token
public_url = ngrok.connect(5000).public_url
print(f"Public URL: {public_url}")

app.run(port=5000)
