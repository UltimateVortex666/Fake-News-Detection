"""
Fake News Detection System
Trains multiple models: BERT, LSTM, and Logistic Regression
"""

import pandas as pd
import numpy as np
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# BERT imports - Using PyTorch for better compatibility
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    try:
        from transformers import AutoTokenizer, TFAutoModel
        PYTORCH_AVAILABLE = False
        print("Warning: PyTorch not available, using TensorFlow BERT (may require tf-keras)")
    except ImportError:
        print("Error: transformers library not properly installed")
        PYTORCH_AVAILABLE = None
import os

# For GloVe embeddings
import gensim.downloader as api

print("Loading datasets...")

# Load datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Add labels: 1 for fake, 0 for true
fake_df['label'] = 1
true_df['label'] = 0

# Combine datasets
df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total articles: {len(df)}")
print(f"Fake news: {df['label'].sum()}")
print(f"True news: {len(df) - df['label'].sum()}")

# Combine title and text for better results
df['text'] = df['title'] + ' ' + df['text']
df = df[['text', 'label']]

# Text preprocessing
def preprocess_text(text):
    """Remove punctuation, lowercase, and clean text"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Preprocessing text...")
df['text'] = df['text'].apply(preprocess_text)

# Remove empty texts
df = df[df['text'].str.len() > 0]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ========== 1. LOGISTIC REGRESSION WITH TF-IDF ==========
print("\n" + "="*50)
print("Training Logistic Regression with TF-IDF...")
print("="*50)

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Predictions
lr_pred = lr_model.predict(X_test_tfidf)
lr_pred_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]

# Evaluate
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_cm = confusion_matrix(y_test, lr_pred)

print(f"Accuracy: {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall: {lr_recall:.4f}")
print(f"F1-Score: {lr_f1:.4f}")
print(f"\nConfusion Matrix:\n{lr_cm}")
print(f"\nClassification Report:\n{classification_report(y_test, lr_pred)}")

# Save model
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("Logistic Regression model saved!")

# ========== 2. LSTM MODEL WITH GLOVE EMBEDDINGS ==========
print("\n" + "="*50)
print("Training LSTM with GloVe embeddings...")
print("="*50)

# Tokenize and pad sequences
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

print("Loading GloVe embeddings...")
try:
    # Try to load pre-trained GloVe embeddings
    glove_model = api.load("glove-wiki-gigaword-100")
    embedding_dim = 100
    print("GloVe embeddings loaded successfully!")
except Exception as e:
    print(f"Could not load GloVe embeddings: {e}")
    print("Using random embeddings instead...")
    glove_model = None
    embedding_dim = 100

# Create embedding matrix
word_index = tokenizer.word_index
embedding_matrix = np.zeros((max_words, embedding_dim))

if glove_model:
    for word, i in word_index.items():
        if i < max_words:
            try:
                embedding_vector = glove_model[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                embedding_matrix[i] = np.random.normal(size=(embedding_dim,))
else:
    # Random initialization
    embedding_matrix = np.random.normal(size=(max_words, embedding_dim))

# Build LSTM model
lstm_model = Sequential([
    Embedding(max_words, embedding_dim, input_length=max_len, 
              weights=[embedding_matrix] if glove_model else None, trainable=True),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("LSTM Model Architecture:")
lstm_model.summary()

# Train LSTM
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = lstm_model.fit(
    X_train_padded, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Predictions
lstm_pred_proba = lstm_model.predict(X_test_padded)
lstm_pred = (lstm_pred_proba > 0.5).astype(int).flatten()

# Evaluate
lstm_accuracy = accuracy_score(y_test, lstm_pred)
lstm_precision = precision_score(y_test, lstm_pred)
lstm_recall = recall_score(y_test, lstm_pred)
lstm_f1 = f1_score(y_test, lstm_pred)
lstm_cm = confusion_matrix(y_test, lstm_pred)

print(f"\nAccuracy: {lstm_accuracy:.4f}")
print(f"Precision: {lstm_precision:.4f}")
print(f"Recall: {lstm_recall:.4f}")
print(f"F1-Score: {lstm_f1:.4f}")
print(f"\nConfusion Matrix:\n{lstm_cm}")
print(f"\nClassification Report:\n{classification_report(y_test, lstm_pred)}")

# Save model
lstm_model.save('lstm_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("LSTM model saved!")

# ========== 3. BERT MODEL ==========
print("\n" + "="*50)
print("Training BERT model...")
print("="*50)

# Load BERT tokenizer and model
model_name = 'distilbert-base-uncased'  # Using DistilBERT for faster training
print(f"Loading {model_name}...")

tokenizer_bert = AutoTokenizer.from_pretrained(model_name)

# Use PyTorch if available (more stable), otherwise TensorFlow
if PYTORCH_AVAILABLE:
    print("Using PyTorch BERT model...")
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.eval()  # Set to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = bert_model.to(device)
    print(f"Using device: {device}")
    
    def encode_texts(texts, tokenizer, max_len=128):
        """Encode texts using BERT tokenizer (PyTorch)"""
        encoded = tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        return encoded
    
    print("Encoding texts with BERT tokenizer...")
    # Process in batches to avoid memory issues
    batch_size = 32
    X_train_bert = []
    X_test_bert = []
    
    # Encode training data
    with torch.no_grad():
        for i in range(0, len(X_train), batch_size):
            batch = X_train.iloc[i:i+batch_size]
            encoded = encode_texts(batch, tokenizer_bert)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = bert_model(**encoded)
            X_train_bert.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())  # [CLS] token
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i+batch_size}/{len(X_train)} training samples")
    
    X_train_bert = np.concatenate(X_train_bert, axis=0)
    print(f"Training embeddings shape: {X_train_bert.shape}")
    
    # Encode test data
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test.iloc[i:i+batch_size]
            encoded = encode_texts(batch, tokenizer_bert)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = bert_model(**encoded)
            X_test_bert.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())  # [CLS] token
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i+batch_size}/{len(X_test)} test samples")
    
    X_test_bert = np.concatenate(X_test_bert, axis=0)
    print(f"Test embeddings shape: {X_test_bert.shape}")
    
else:
    # TensorFlow fallback (requires tf-keras)
    print("Using TensorFlow BERT model...")
    try:
        bert_model = TFAutoModel.from_pretrained(model_name)
        
        def encode_texts(texts, tokenizer, max_len=128):
            """Encode texts using BERT tokenizer (TensorFlow)"""
            encoded = tokenizer(
                texts.tolist(),
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='tf'
            )
            return encoded
        
        print("Encoding texts with BERT tokenizer...")
        batch_size = 32
        X_train_bert = []
        X_test_bert = []
        
        # Encode training data
        for i in range(0, len(X_train), batch_size):
            batch = X_train.iloc[i:i+batch_size]
            encoded = encode_texts(batch, tokenizer_bert)
            X_train_bert.append(bert_model(encoded)[0][:, 0, :])  # [CLS] token
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i+batch_size}/{len(X_train)} training samples")
        
        X_train_bert = tf.concat(X_train_bert, axis=0).numpy()
        print(f"Training embeddings shape: {X_train_bert.shape}")
        
        # Encode test data
        for i in range(0, len(X_test), batch_size):
            batch = X_test.iloc[i:i+batch_size]
            encoded = encode_texts(batch, tokenizer_bert)
            X_test_bert.append(bert_model(encoded)[0][:, 0, :])  # [CLS] token
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i+batch_size}/{len(X_test)} test samples")
        
        X_test_bert = tf.concat(X_test_bert, axis=0).numpy()
        print(f"Test embeddings shape: {X_test_bert.shape}")
        
    except Exception as e:
        print(f"Error loading TensorFlow BERT: {e}")
        print("Please install tf-keras: pip install tf-keras")
        print("Or install PyTorch: pip install torch")
        raise

# Standardize BERT embeddings (already numpy arrays if PyTorch was used)
scaler_bert = StandardScaler()
X_train_bert_scaled = scaler_bert.fit_transform(X_train_bert)
X_test_bert_scaled = scaler_bert.transform(X_test_bert)

# Build classifier on top of BERT embeddings
bert_classifier = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_bert_scaled.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

bert_classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("BERT Classifier Architecture:")
bert_classifier.summary()

# Train BERT classifier
history_bert = bert_classifier.fit(
    X_train_bert_scaled, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
    verbose=1
)

# Predictions
bert_pred_proba = bert_classifier.predict(X_test_bert_scaled)
bert_pred = (bert_pred_proba > 0.5).astype(int).flatten()

# Evaluate
bert_accuracy = accuracy_score(y_test, bert_pred)
bert_precision = precision_score(y_test, bert_pred)
bert_recall = recall_score(y_test, bert_pred)
bert_f1 = f1_score(y_test, bert_pred)
bert_cm = confusion_matrix(y_test, bert_pred)

print(f"\nAccuracy: {bert_accuracy:.4f}")
print(f"Precision: {bert_precision:.4f}")
print(f"Recall: {bert_recall:.4f}")
print(f"F1-Score: {bert_f1:.4f}")
print(f"\nConfusion Matrix:\n{bert_cm}")
print(f"\nClassification Report:\n{classification_report(y_test, bert_pred)}")

# Save BERT model
bert_classifier.save('bert_classifier.h5')
tokenizer_bert.save_pretrained('bert_tokenizer')
if PYTORCH_AVAILABLE:
    bert_model.save_pretrained('bert_model')
    # Save PyTorch flag
    with open('bert_use_pytorch.pkl', 'wb') as f:
        pickle.dump(True, f)
else:
    bert_model.save_pretrained('bert_model')
    with open('bert_use_pytorch.pkl', 'wb') as f:
        pickle.dump(False, f)
with open('bert_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_bert, f)

print("BERT model saved!")

# ========== FALSE POSITIVE/NEGATIVE ANALYSIS ==========
print("\n" + "="*50)
print("FALSE POSITIVE/NEGATIVE ANALYSIS")
print("="*50)

def analyze_false_predictions(y_true, y_pred, X_test_data, model_name):
    """Analyze false positives and false negatives"""
    false_positives = X_test_data[(y_true == 0) & (y_pred == 1)]
    false_negatives = X_test_data[(y_true == 1) & (y_pred == 0)]
    
    print(f"\n{model_name} - False Positives (Predicted Fake but Actually True): {len(false_positives)}")
    print(f"{model_name} - False Negatives (Predicted True but Actually Fake): {len(false_negatives)}")
    
    if len(false_positives) > 0:
        print(f"\nSample False Positives ({model_name}):")
        for idx, text in enumerate(false_positives.head(3)):
            print(f"{idx+1}. {text[:200]}...")
    
    if len(false_negatives) > 0:
        print(f"\nSample False Negatives ({model_name}):")
        for idx, text in enumerate(false_negatives.head(3)):
            print(f"{idx+1}. {text[:200]}...")
    
    return false_positives, false_negatives

# Analyze for each model
lr_fp, lr_fn = analyze_false_predictions(y_test, lr_pred, X_test, "Logistic Regression")
lstm_fp, lstm_fn = analyze_false_predictions(y_test, lstm_pred, X_test, "LSTM")
bert_fp, bert_fn = analyze_false_predictions(y_test, bert_pred, X_test, "BERT")

# ========== MODEL COMPARISON ==========
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)

comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'LSTM', 'BERT'],
    'Accuracy': [lr_accuracy, lstm_accuracy, bert_accuracy],
    'Precision': [lr_precision, lstm_precision, bert_precision],
    'Recall': [lr_recall, lstm_recall, bert_recall],
    'F1-Score': [lr_f1, lstm_f1, bert_f1]
})

print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv('model_comparison.csv', index=False)
print("\nModel comparison saved to model_comparison.csv")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print("\nAll models have been trained and saved.")
print("You can now use the web interface to make predictions!")

