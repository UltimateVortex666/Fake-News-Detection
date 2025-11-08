# Fix for Keras 3 Compatibility Issue

## Problem
The error occurred because Keras 3 is not compatible with TensorFlow transformers models:
```
ValueError: Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers.
```

## Solution
The code has been updated to use **PyTorch BERT** instead of TensorFlow BERT, which avoids the Keras 3 compatibility issue entirely.

## Changes Made

### 1. train_models.py
- Added PyTorch support with automatic fallback to TensorFlow
- BERT model now uses PyTorch by default (more stable)
- Saves a flag to indicate which backend was used

### 2. app.py
- Updated to load PyTorch or TensorFlow BERT based on saved flag
- Handles both backends seamlessly

### 3. predict.py
- Updated to support both PyTorch and TensorFlow BERT

### 4. requirements.txt
- Added `torch>=2.0.0` for PyTorch support

## Installation

Install PyTorch (if not already installed):
```bash
pip install torch
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Benefits of PyTorch BERT

1. **No Keras 3 compatibility issues** - Works with any Keras version
2. **Better stability** - PyTorch transformers are more mature
3. **GPU support** - Automatically uses GPU if available
4. **Faster inference** - PyTorch models are optimized

## Running the Code

After installing PyTorch, simply run:
```bash
python train_models.py
```

The script will automatically detect PyTorch and use it for BERT. If PyTorch is not available, it will fall back to TensorFlow (but you'll need to install `tf-keras` for that).

## Note

The LSTM and Logistic Regression models are unchanged and continue to work as before. Only the BERT model backend has been updated.

