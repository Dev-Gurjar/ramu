import joblib
import streamlit as st 
import numpy as np 
import os
from PIL import Image
import pandas as pd

# Load models
svm_model = joblib.load('svm_model.pkl')
naive_bayes_model = joblib.load('NB_model.pkl')
pca = joblib.load('pca_model.pkl')
lda = joblib.load('lda_model.pkl')

st.title('Streamlit Example')

# Function to load a single image from a file-path
def load_img(file_path, slice_, color, resize):
    default_slice = (slice(0, 250), slice(0, 250))  # Default slice size

    # Use the provided slice or the default
    if slice_ is None: 
        slice_ = default_slice
    else: 
        slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    # Calculate the height and width from the slice
    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    # Apply resizing if needed
    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)

    # Initialize the image array
    if not color: 
        face = np.zeros((h, w), dtype=np.float32)
    else: 
        face = np.zeros((h, w, 3), dtype=np.float32)

    # Load and process the image
    pil_img = Image.open(file_path)
    pil_img = pil_img.crop((w_slice.start, h_slice.start, w_slice.stop, h_slice.stop))

    if resize is not None: 
        pil_img = pil_img.resize((w, h))
    face = np.asarray(pil_img, dtype=np.float32)

    # Normalize pixel values and convert to grayscale if not color
    face /= 255.0
    if not color: 
        face = face.mean(axis=2)

    return face

# Create the temp directory if it doesn't exist
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    with open(os.path.join(temp_dir, "temp_image.jpg"), "wb") as f:
        f.write(file.read())
    
    image_path = os.path.join(temp_dir, "temp_image.jpg")
    image = Image.open(image_path)
    st.image(image, use_column_width=True)
    
    face = load_img(image_path, slice_=None, color=False, resize=0.4)

    X = face.reshape(1, -1)
    X_t = pca.transform(X)
    X_t = lda.transform(X_t)
    
    # Perform predictions
    naive_bayes_prediction = naive_bayes_model.predict(X_t)
    svm_prediction = svm_model.predict(X_t)

    # Load target data
    names = pd.read_csv("target_data.csv")

    # Display predictions
    st.write("Naive Bayes Prediction:", names["person_name"][naive_bayes_prediction])
    st.write("SVM Prediction:", names["person_name"][svm_prediction])
