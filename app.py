"""
Fake News Detection Web Interface
Streamlit app for predicting fake news
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
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
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid;
    }
    .fake-prediction {
        background-color: #ffebee;
        border-color: #f44336;
    }
    .real-prediction {
        background-color: #e8f5e9;
        border-color: #4caf50;
    }
    .model-result {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    try:
        # Load Logistic Regression model
        with open('lr_model.pkl', 'rb') as f:
            models['lr'] = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            models['tfidf'] = pickle.load(f)
        st.success("✓ Logistic Regression model loaded")
    except Exception as e:
        st.error(f"Error loading Logistic Regression model: {e}")
        models['lr'] = None
        models['tfidf'] = None
    
    try:
        # Load LSTM model
        models['lstm'] = load_model('lstm_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            models['tokenizer'] = pickle.load(f)
        st.success("✓ LSTM model loaded")
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")
        models['lstm'] = None
        models['tokenizer'] = None
    
    try:
        # Load BERT model
        models['bert_classifier'] = load_model('bert_classifier.h5')
        models['bert_tokenizer'] = AutoTokenizer.from_pretrained('bert_tokenizer')
        
        # Check if PyTorch or TensorFlow was used
        try:
            with open('bert_use_pytorch.pkl', 'rb') as f:
                use_pytorch = pickle.load(f)
        except:
            # Default to PyTorch if file doesn't exist
            use_pytorch = PYTORCH_AVAILABLE
        
        if use_pytorch and PYTORCH_AVAILABLE:
            models['bert_model'] = AutoModel.from_pretrained('bert_model')
            models['bert_model'].eval()
            models['bert_device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            models['bert_model'] = models['bert_model'].to(models['bert_device'])
        else:
            if not PYTORCH_AVAILABLE:
                models['bert_model'] = TFAutoModel.from_pretrained('bert_model')
            else:
                # Try to load as PyTorch anyway
                models['bert_model'] = AutoModel.from_pretrained('bert_model')
                models['bert_model'].eval()
                models['bert_device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                models['bert_model'] = models['bert_model'].to(models['bert_device'])
                use_pytorch = True
        
        models['bert_use_pytorch'] = use_pytorch
        with open('bert_scaler.pkl', 'rb') as f:
            models['bert_scaler'] = pickle.load(f)
        st.success("✓ BERT model loaded")
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        models['bert_classifier'] = None
        models['bert_tokenizer'] = None
        models['bert_model'] = None
        models['bert_scaler'] = None
        models['bert_use_pytorch'] = False
    
    return models

def preprocess_text(text):
    """Preprocess text for prediction"""
    if pd.isna(text) or text == "":
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

def predict_lr(text, model, vectorizer):
    """Predict using Logistic Regression"""
    if model is None or vectorizer is None:
        return None, None
    
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0][1]
    
    return prediction, probability

def predict_lstm(text, model, tokenizer):
    """Predict using LSTM"""
    if model is None or tokenizer is None:
        return None, None
    
    processed_text = preprocess_text(text)
    max_len = 200
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # Predict
    probability = model.predict(padded, verbose=0)[0][0]
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def predict_bert(text, model, tokenizer, bert_model, scaler, use_pytorch=False, device=None):
    """Predict using BERT"""
    if model is None or tokenizer is None or bert_model is None or scaler is None:
        return None, None
    
    processed_text = preprocess_text(text)
    
    if use_pytorch and device is not None:
        # PyTorch version
        encoded = tokenizer(
            processed_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = bert_model(**encoded)
            bert_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
    else:
        # TensorFlow version
        encoded = tokenizer(
            processed_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='tf'
        )
        # Get BERT embeddings
        bert_embeddings = bert_model(encoded)[0][:, 0, :].numpy()  # [CLS] token
    
    # Scale embeddings
    bert_embeddings_scaled = scaler.transform(bert_embeddings)
    
    # Predict
    probability = model.predict(bert_embeddings_scaled, verbose=0)[0][0]
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Classify news articles as Real or Fake using AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Predict", "About", "Model Performance"])
    
    if page == "Predict":
        # Load models
        with st.spinner("Loading models... This may take a moment on first run."):
            models = load_models()
        
        st.markdown("---")
        
        # Input section
        st.subheader("Enter News Article Text")
        
        input_method = st.radio("Input method", ["Type text", "Upload file"])
        
        text_input = ""
        
        if input_method == "Type text":
            text_input = st.text_area(
                "News Article Text",
                height=200,
                placeholder="Paste or type the news article text here..."
            )
        else:
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            if uploaded_file is not None:
                text_input = str(uploaded_file.read(), "utf-8")
                st.text_area("File content", text_input, height=200)
        
        # Predict button
        if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
            if text_input.strip() == "":
                st.error("Please enter some text to analyze!")
            else:
                # Show predictions
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                results = []
                
                # Logistic Regression
                if models['lr'] is not None:
                    lr_pred, lr_prob = predict_lr(text_input, models['lr'], models['tfidf'])
                    if lr_pred is not None:
                        results.append({
                            'Model': 'Logistic Regression',
                            'Prediction': 'Fake' if lr_pred == 1 else 'Real',
                            'Confidence': f"{lr_prob*100:.2f}%",
                            'Probability': lr_prob
                        })
                
                # LSTM
                if models['lstm'] is not None:
                    lstm_pred, lstm_prob = predict_lstm(text_input, models['lstm'], models['tokenizer'])
                    if lstm_pred is not None:
                        results.append({
                            'Model': 'LSTM',
                            'Prediction': 'Fake' if lstm_pred == 1 else 'Real',
                            'Confidence': f"{lstm_prob*100:.2f}%",
                            'Probability': lstm_prob
                        })
                
                # BERT
                if models['bert_classifier'] is not None:
                    bert_use_pytorch = models.get('bert_use_pytorch', False)
                    bert_device = models.get('bert_device', None)
                    bert_pred, bert_prob = predict_bert(
                        text_input, 
                        models['bert_classifier'], 
                        models['bert_tokenizer'],
                        models['bert_model'],
                        models['bert_scaler'],
                        use_pytorch=bert_use_pytorch,
                        device=bert_device
                    )
                    if bert_pred is not None:
                        results.append({
                            'Model': 'BERT',
                            'Prediction': 'Fake' if bert_pred == 1 else 'Real',
                            'Confidence': f"{bert_prob*100:.2f}%",
                            'Probability': bert_prob
                        })
                
                if results:
                    # Display results
                    df_results = pd.DataFrame(results)
                    
                    # Average prediction
                    avg_prob = df_results['Probability'].mean()
                    avg_pred = 'Fake' if avg_prob > 0.5 else 'Real'
                    
                    # Display average prediction prominently
                    if avg_pred == 'Fake':
                        st.markdown(f"""
                            <div class="prediction-box fake-prediction">
                                <h2>⚠️ Overall Prediction: FAKE NEWS</h2>
                                <h3>Average Confidence: {avg_prob*100:.2f}%</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="prediction-box real-prediction">
                                <h2>✅ Overall Prediction: REAL NEWS</h2>
                                <h3>Average Confidence: {avg_prob*100:.2f}%</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Individual model results
                    st.subheader("Individual Model Predictions")
                    for _, row in df_results.iterrows():
                        pred_class = "fake-prediction" if row['Prediction'] == 'Fake' else "real-prediction"
                        st.markdown(f"""
                            <div class="model-result">
                                <h4>{row['Model']}</h4>
                                <p><strong>Prediction:</strong> {row['Prediction']}</p>
                                <p><strong>Confidence:</strong> {row['Confidence']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Results table
                    st.subheader("Detailed Results")
                    st.dataframe(df_results[['Model', 'Prediction', 'Confidence']], use_container_width=True)
                    
                    # Visualization
                    st.subheader("Confidence Scores Visualization")
                    chart_data = pd.DataFrame({
                        'Model': df_results['Model'],
                        'Fake Probability': df_results['Probability'] * 100,
                        'Real Probability': (1 - df_results['Probability']) * 100
                    })
                    st.bar_chart(chart_data.set_index('Model'))
                else:
                    st.error("No models were loaded successfully. Please train the models first using train_models.py")
    
    elif page == "About":
        st.header("About Fake News Detection System")
        st.markdown("""
        This application uses three different machine learning models to classify news articles as **Real** or **Fake**:
        
        ### Models Used:
        
        1. **Logistic Regression with TF-IDF**
           - Traditional machine learning approach
           - Fast and interpretable
           - Uses TF-IDF vectorization for text features
        
        2. **LSTM (Long Short-Term Memory)**
           - Deep learning model with GloVe embeddings
           - Captures sequential patterns in text
           - Bidirectional LSTM for better context understanding
        
        3. **BERT (Bidirectional Encoder Representations from Transformers)**
           - State-of-the-art transformer model
           - Uses DistilBERT for faster inference
           - Best at understanding context and semantics
        
        ### How to Use:
        
        1. Go to the **Predict** page
        2. Enter or upload the news article text
        3. Click "Analyze Article"
        4. View predictions from all three models
        
        ### Training:
        
        Before using this app, make sure to train the models by running:
        ```bash
        python train_models.py
        ```
        
        This will train all three models on the Fake.csv and True.csv datasets.
        """)
    
    elif page == "Model Performance":
        st.header("Model Performance Metrics")
        
        try:
            # Try to load comparison CSV
            comparison = pd.read_csv('model_comparison.csv')
            st.dataframe(comparison, use_container_width=True)
            
            # Visualization
            st.subheader("Performance Comparison")
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            fig_data = comparison.set_index('Model')[metrics]
            st.bar_chart(fig_data)
            
        except FileNotFoundError:
            st.warning("Model performance data not found. Please run train_models.py first to generate performance metrics.")
            st.info("""
            The training script will generate:
            - Accuracy scores for each model
            - Precision and Recall metrics
            - F1-Score comparisons
            - Confusion matrices
            - False positive/negative analysis
            """)

if __name__ == "__main__":
    main()

