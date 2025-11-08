"""
Command-line prediction script for fake news detection
Usage: python predict.py "Your news article text here"
"""

import sys
import pickle
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Try PyTorch first, fallback to TensorFlow
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    from transformers import AutoTokenizer, TFAutoModel
    PYTORCH_AVAILABLE = False
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    """Preprocess text for prediction"""
    if pd.isna(text) or text == "":
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_lr(text):
    """Predict using Logistic Regression"""
    try:
        with open('lr_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        
        processed_text = preprocess_text(text)
        text_vector = tfidf.transform([processed_text])
        probability = lr_model.predict_proba(text_vector)[0][1]
        prediction = 'Fake' if lr_model.predict(text_vector)[0] == 1 else 'Real'
        return prediction, probability
    except Exception as e:
        return None, None

def predict_lstm(text):
    """Predict using LSTM"""
    try:
        lstm_model = load_model('lstm_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        processed_text = preprocess_text(text)
        max_len = 200
        
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        
        probability = lstm_model.predict(padded, verbose=0)[0][0]
        prediction = 'Fake' if probability > 0.5 else 'Real'
        return prediction, probability
    except Exception as e:
        return None, None

def predict_bert(text):
    """Predict using BERT"""
    try:
        bert_classifier = load_model('bert_classifier.h5')
        bert_tokenizer = AutoTokenizer.from_pretrained('bert_tokenizer')
        
        # Check if PyTorch or TensorFlow was used
        try:
            with open('bert_use_pytorch.pkl', 'rb') as f:
                use_pytorch = pickle.load(f)
        except:
            use_pytorch = PYTORCH_AVAILABLE
        
        if use_pytorch and PYTORCH_AVAILABLE:
            bert_model = AutoModel.from_pretrained('bert_model')
            bert_model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            bert_model = bert_model.to(device)
        else:
            if PYTORCH_AVAILABLE:
                bert_model = AutoModel.from_pretrained('bert_model')
                bert_model.eval()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                bert_model = bert_model.to(device)
                use_pytorch = True
            else:
                bert_model = TFAutoModel.from_pretrained('bert_model')
                use_pytorch = False
                device = None
        
        with open('bert_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        processed_text = preprocess_text(text)
        
        if use_pytorch and device is not None:
            # PyTorch version
            encoded = bert_tokenizer(
                processed_text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = bert_model(**encoded)
                bert_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        else:
            # TensorFlow version
            encoded = bert_tokenizer(
                processed_text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='tf'
            )
            bert_embeddings = bert_model(encoded)[0][:, 0, :].numpy()
        
        bert_embeddings_scaled = scaler.transform(bert_embeddings)
        
        probability = bert_classifier.predict(bert_embeddings_scaled, verbose=0)[0][0]
        prediction = 'Fake' if probability > 0.5 else 'Real'
        return prediction, probability
    except Exception as e:
        print(f"BERT prediction error: {e}")
        return None, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your news article text here\"")
        sys.exit(1)
    
    text = ' '.join(sys.argv[1:])
    
    print("="*60)
    print("FAKE NEWS DETECTION - PREDICTION RESULTS")
    print("="*60)
    print(f"\nInput Text: {text[:200]}...")
    print("\n" + "-"*60)
    
    results = []
    
    # Logistic Regression
    lr_pred, lr_prob = predict_lr(text)
    if lr_pred:
        results.append(('Logistic Regression', lr_pred, lr_prob))
        print(f"Logistic Regression: {lr_pred} (Confidence: {lr_prob*100:.2f}%)")
    
    # LSTM
    lstm_pred, lstm_prob = predict_lstm(text)
    if lstm_pred:
        results.append(('LSTM', lstm_pred, lstm_prob))
        print(f"LSTM: {lstm_pred} (Confidence: {lstm_prob*100:.2f}%)")
    
    # BERT
    bert_pred, bert_prob = predict_bert(text)
    if bert_pred:
        results.append(('BERT', bert_pred, bert_prob))
        print(f"BERT: {bert_pred} (Confidence: {bert_prob*100:.2f}%)")
    
    if not results:
        print("Error: No models found. Please train models first using train_models.py")
        sys.exit(1)
    
    # Average prediction
    avg_prob = np.mean([prob for _, _, prob in results])
    avg_pred = 'Fake' if avg_prob > 0.5 else 'Real'
    
    print("\n" + "-"*60)
    print(f"OVERALL PREDICTION: {avg_pred}")
    print(f"Average Confidence: {avg_prob*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()

