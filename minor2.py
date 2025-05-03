import streamlit as st
import cv2
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_crack_model():
    model = load_model('best_model.h5')  # Make sure this matches your notebook save path
    return model

model = load_crack_model()

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel if present
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Crack analysis function
def analyze_and_display_crack(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crack_image = image.copy()
    crack_found = False

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            crack_found = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(crack_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(crack_image, f"Width: {w}px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(crack_image, f"Height: {h}px", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return crack_image, crack_found

# Streamlit UI
st.title("Real-Time Crack Detection on Concrete Surfaces")

image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img_array = preprocess_image(image)
    
    prediction = model.predict(img_array)[0][0]
    label = "Uncracked" if prediction > 0.2 else "Cracked"

    st.markdown(f"### Prediction: {label} ({prediction:.2f})")

    # Convert to OpenCV image for analysis
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    analyzed_image, crack_found = analyze_and_display_crack(image_cv)

    st.image(cv2.cvtColor(analyzed_image, cv2.COLOR_BGR2RGB), caption="Crack Analysis Output")

    if not crack_found:
        st.markdown("**No significant crack detected.**")
