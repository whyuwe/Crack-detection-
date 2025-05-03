import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('crack_detection_model.h5')
class_names = ['not_cracked', 'cracked']

# Streamlit interface
st.title("Real-Time Crack Detection")

# Streamlit webcam input
st.write("to start real-time crack detection using .")

# Function to preprocess and make predictions
def predict_crack(frame):
    img = cv2.resize(frame, (224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0][0]
    label = class_names[int(prediction > 0.5)]
    return label, prediction

# Streamlit webcam video capture (via st.camera_input)
camera_input = st.camera_input("Take a picture")

if camera_input:
    # Read the uploaded image file and process it
    image = Image.open(camera_input)
    frame = np.array(image)
    
    # Prediction
    label, prediction = predict_crack(frame)
    
    # Display results
    st.image(frame, caption="Uploaded Image", use_column_width=True)
    st.write(f"Prediction: {label} with confidence: {prediction:.2f}")

    # Provide a real-time feedback button (if you want to reset or reload)
    if st.button('Clear'):
        st.experimental_rerun()

