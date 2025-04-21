import numpy as np
import streamlit as st
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

# Ignore warnings in output
import warnings
warnings.filterwarnings("ignore")

# Load model and define word dictionary
word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
             19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Load the pre-trained model
try:
    model = load_model('modelHandWritten.h5')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Function to preprocess and classify the image
def classify(img):
    try:
        # Apply Gaussian blur
        img = cv2.GaussianBlur(img, (7, 7), 0)
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Resize to 28x28
        img_final = cv2.resize(img_thresh, (28, 28))
        
        # Reshape for model input
        img_final = np.reshape(img_final, (1, 28, 28, 1))
        
        # Predict the character
        prediction = model.predict(img_final, verbose=0)
        img_pred = word_dict[np.argmax(prediction)]
        
        return img_pred
    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
        return None

# Streamlit UI
st.title("Handwritten Letter Recognition")
st.write("Draw a letter in the box below or upload an image, and the model will predict the letter.")

# File uploader for external images
uploaded_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Classify the uploaded image
    prediction = classify(img)
    
    if prediction:
        st.write(f"Prediction: **{prediction}**")