# Handwritten Digit Recognizer

This project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to recognize handwritten digits (0-9).

## Project Overview

- **Dataset:** MNIST (Modified National Institute of Standards and Technology) dataset containing 70,000 grayscale images of digits (28x28 pixels).
- **Architecture:** Convolutional Neural Network (CNN) with:
    - 2 Convolutional layers
    - 2 Max-pooling layers
    - Flattening and Dense layers for classification.
- **Frontend:** Streamlit web application for real-time inference on uploaded images.

## Setup Instructions

### 1. Installation

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (Linux/Mac)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Model

Before running the app, you must train the neural network:

```bash
python train.py
```
This will:
- Download the MNIST dataset.
- Train the model for 5 epochs.
- Save the model to `models/digit_recognizer_model.h5`.
- Save a training plot as `training_metrics.png`.

### 3. Launching the App

Once trained, start the Streamlit web interface:

```bash
streamlit run app.py
```
Upload any image (JPG/PNG) containing a single handwritten digit to see the results.

## Model Performance

The CNN architecture typically achieves **~99% accuracy** on the MNIST test set after just a few epochs of training.

## Future Improvements

- Add a drawing canvas for users to sketch digits directly.
- Deploy the model as a REST API using FastAPI.
- Implement more advanced architectures (e.g., ResNet or Transformers).
