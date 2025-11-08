# Fake News Detection System

A comprehensive NLP-based system to classify news articles as real or fake using multiple machine learning models: BERT, LSTM, and Logistic Regression.

## Features

- **Three ML Models**: Logistic Regression, LSTM, and BERT for classification
- **Multiple Embeddings**: TF-IDF, GloVe, and BERT embeddings
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- **False Positive/Negative Analysis**: Detailed analysis of misclassified articles
- **Web Interface**: User-friendly Streamlit web app for real-time predictions

## Dataset

The system uses two CSV files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains real news articles

Both files should have columns: `title`, `text`, `subject`, `date`

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Note: The first run will download:
   - GloVe embeddings (if available)
   - BERT model weights
   - This may take some time depending on your internet connection

## Usage

### Step 1: Train the Models

Train all three models on your dataset:
```bash
python train_models.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression with TF-IDF
- Train LSTM with GloVe embeddings
- Train BERT classifier
- Evaluate all models
- Save trained models and tokenizers
- Generate performance comparison

**Training time**: Approximately 30-60 minutes depending on your hardware and dataset size.

### Step 2: Run the Web Interface

Start the Streamlit web application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Model Details

### 1. Logistic Regression (TF-IDF)
- **Features**: TF-IDF vectorization with 5000 features
- **Advantages**: Fast, interpretable, good baseline
- **Use Case**: Quick predictions with decent accuracy

### 2. LSTM with GloVe Embeddings
- **Architecture**: Bidirectional LSTM with 2 layers
- **Embeddings**: GloVe (100-dimensional)
- **Advantages**: Captures sequential patterns in text
- **Use Case**: Better understanding of context and word order

### 3. BERT Classifier
- **Model**: DistilBERT-base-uncased
- **Architecture**: BERT embeddings + Dense classifier
- **Advantages**: State-of-the-art performance, best context understanding
- **Use Case**: Highest accuracy predictions

## Evaluation Metrics

The training script generates:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **False Positive/Negative Analysis**: Examples of misclassified articles

## Output Files

After training, the following files will be created:

### Models:
- `lr_model.pkl`: Logistic Regression model
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer
- `lstm_model.h5`: LSTM model
- `tokenizer.pkl`: Text tokenizer for LSTM
- `bert_classifier.h5`: BERT classifier
- `bert_tokenizer/`: BERT tokenizer directory
- `bert_model/`: BERT model directory
- `bert_scaler.pkl`: Scaler for BERT embeddings

### Evaluation:
- `model_comparison.csv`: Performance comparison of all models

## Web Interface Features

1. **Predict Page**: 
   - Enter news article text
   - Upload text file
   - Get predictions from all three models
   - View confidence scores
   - See overall prediction

2. **About Page**: 
   - Information about the system
   - Model descriptions
   - Usage instructions

3. **Model Performance Page**: 
   - View performance metrics
   - Compare model accuracies
   - Visualization of results

## Text Preprocessing

The system performs the following preprocessing steps:
- Remove URLs
- Remove special characters and digits
- Convert to lowercase
- Remove extra whitespace
- Tokenization (for LSTM and BERT)

## Troubleshooting

### Memory Issues
If you encounter memory errors:
- Reduce batch size in `train_models.py`
- Use a smaller sample of the dataset
- Process data in smaller chunks

### Model Loading Errors
If models fail to load:
- Ensure all models are trained first
- Check that all model files are in the same directory
- Verify file permissions

### GloVe Download Issues
If GloVe embeddings fail to download:
- The script will automatically use random embeddings
- Performance may be slightly reduced
- You can manually download GloVe embeddings

## Performance Tips

1. **GPU Acceleration**: Use GPU for faster BERT and LSTM training
2. **Batch Processing**: Adjust batch sizes based on available memory
3. **Model Selection**: Use BERT for best accuracy, Logistic Regression for speed

## Future Improvements

- Add more models (RoBERTa, XLNet)
- Implement ensemble voting
- Add model explanation/interpretability
- Support for multiple languages
- Real-time training with new data

## License

This project is open source and available for educational purposes.

## Contact

For questions or issues, please check the code comments or create an issue in the repository.

