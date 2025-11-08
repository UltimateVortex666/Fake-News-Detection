# Quick Start Guide

## Step-by-Step Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This may take 10-15 minutes as it installs TensorFlow, Transformers, and other large packages.

### 2. Verify Dataset Files

Make sure you have these files in the project directory:
- `Fake.csv`
- `True.csv`

### 3. Train the Models

Run the training script:

```bash
python train_models.py
```

**Important Notes**:
- This will take 30-60 minutes depending on your hardware
- You need at least 8GB RAM (16GB recommended)
- The script will download BERT model weights (~250MB) on first run
- GloVe embeddings will be downloaded automatically if available

**What happens during training**:
1. Data loading and preprocessing
2. Training Logistic Regression (fast, ~2-5 minutes)
3. Training LSTM with GloVe (medium, ~10-20 minutes)
4. Training BERT classifier (slow, ~20-40 minutes)
5. Model evaluation and saving

### 4. Run the Web Interface

Once training is complete, start the web app:

```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`

### 5. Make Predictions

1. Click on **Predict** in the sidebar
2. Enter news article text or upload a file
3. Click **Analyze Article**
4. View predictions from all three models

## Alternative: Command Line Prediction

You can also make predictions from the command line:

```bash
python predict.py "Your news article text here"
```

## Troubleshooting

### Out of Memory Error

If you get memory errors:
1. Reduce batch size in `train_models.py` (line ~250, change `batch_size=32` to `batch_size=16`)
2. Process a smaller dataset sample
3. Close other applications

### Models Not Found

If the web app says "models not found":
- Make sure you've run `train_models.py` first
- Check that all model files are in the same directory
- Verify files: `lr_model.pkl`, `lstm_model.h5`, `bert_classifier.h5`, etc.

### Slow Performance

- Use GPU if available (automatic with TensorFlow)
- Reduce max_length for BERT (line ~242 in train_models.py)
- Use fewer epochs for faster training

## Expected Output Files

After training, you should see:
- `lr_model.pkl`
- `tfidf_vectorizer.pkl`
- `lstm_model.h5`
- `tokenizer.pkl`
- `bert_classifier.h5`
- `bert_tokenizer/` (directory)
- `bert_model/` (directory)
- `bert_scaler.pkl`
- `model_comparison.csv`

## Next Steps

- Experiment with different text preprocessing
- Try ensemble methods
- Add more training data
- Fine-tune hyperparameters

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Review error messages in the console
3. Verify all dependencies are installed correctly

