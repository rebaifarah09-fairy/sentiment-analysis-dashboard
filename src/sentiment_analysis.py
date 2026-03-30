from transformers import pipeline
import torch

def load_sentiment_model():
    """Charge un modèle léger et rapide"""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",   # Modèle léger
        device=device
    )

def analyze_text(text, classifier):
    """Analyse le sentiment d'un texte"""
    if not text or len(str(text).strip()) < 3:
        return "NEUTRAL", 0.50
    
    result = classifier(str(text)[:512])[0]
    
    label = result['label']
    score = result['score']
    
    if label == "POSITIVE" or label == "LABEL_1":
        return "POSITIVE", round(score, 4)
    else:
        return "NEGATIVE", round(score, 4)