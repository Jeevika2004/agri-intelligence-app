import streamlit as st
import numpy as np
from keras.models import load_model # type: ignore
from keras.preprocessing import image # type: ignore
from PIL import Image
import os

# Load the trained model
model = load_model("leaf_disease_model.h5")  # adjust path as needed

# 👉 Debug: Print model's expected input shape
st.write("Model expects input shape:", model.input_shape)


# Define the class labels (use your own labels here)
data_dir = "data/archive/plantvillage dataset/color"
class_labels = sorted(os.listdir(data_dir))


# UI
st.title("🍃 Leaf Disease Detector")
uploaded_file = st.file_uploader("Upload a plant leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prepare image
    img = Image.open(uploaded_file).resize((128, 128))  # Resize to what the model expects
    st.image(img, caption="Uploaded Image", use_container_width=True)  # update deprecated param

    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)
  


    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    st.success(f"🔍 Predicted Disease: **{predicted_class}**")



