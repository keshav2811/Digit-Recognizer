import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# Set page config
st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="🔢")

st.title("🔢 Handwritten Digit Recognizer")
st.markdown("""
Upload an image of a handwritten digit (0-9) to see the model's prediction.
The image should ideally be a single digit on a clear background.
""")

# Load the model
MODEL_PATH = 'models/digit_recognizer_model.h5'

@st.cache_resource
def load_digit_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("⚠️ Model not found! Please run `python train.py` first to train the model.")
        return None

model = load_digit_model()

def preprocess_image(img):
    # Convert to grayscale
    img = img.convert('L')
    
    # Invert colors (MNIST is white digits on black background)
    # Most user uploads are black digits on white background
    img = ImageOps.invert(img)
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Reshape for the model (Batch size, Height, Width, Channels)
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if model is not None:
        processed_img = preprocess_image(image)
        
        # Prediction
        prediction = model.predict(processed_img)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        with col2:
            st.subheader("Prediction")
            st.markdown(f"### Predicted Digit: **{predicted_digit}**")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            
            # Show probability bar chart
            st.bar_chart(prediction[0])
            st.caption("Probability distribution across digits 0-9")
    else:
        st.error("Model is not loaded. Ensure you've trained the model successfully.")

st.sidebar.title("About")
st.sidebar.info(
    "This application uses a Convolutional Neural Network (CNN) trained on the "
    "MNIST dataset of 70,000 handwritten digits. "
    "Built with TensorFlow and Keras."
)
