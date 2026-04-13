import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Set page config
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢", layout="centered")

@st.cache_resource
def load_model():
    return keras.models.load_model('mnist_model.keras')

@st.cache_data
def load_data():
    df = pd.read_csv('mnist_test.csv')
    return df

st.title("🔢 MNIST Digit Recognizer")
st.write("Upload an image of a handwritten digit or select a random sample from the test dataset to see the model's prediction!")

try:
    with st.spinner("Loading model and data..."):
        model = load_model()
        df = load_data()
except Exception as e:
    st.error(f"Error loading model or data. Please ensure 'mnist_model.keras' and 'mnist_test.csv' exist. Error: {str(e)}")
    st.stop()

tab1, tab2 = st.tabs(["🖼️ Upload Image", "🎲 Random Test Sample"])

with tab1:
    st.header("Upload a Digit Image")
    st.write("For best results, upload an image of a single digit on a plain background.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L') # Convert to grayscale
        st.image(image, caption='Uploaded Image', width=150)
        
        # Preprocess the image
        img_resized = image.resize((28, 28))
        img_array = np.array(img_resized)
        
        # Invert colors if necessary (MNIST is white text on black background)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
            
        img_array = img_array / 255.0
        img_flattened = img_array.flatten().reshape(1, 784)
        
        if st.button("Predict Uploaded Image"):
            prediction = model.predict(img_flattened)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            st.write(f"### 🎯 Predicted Digit: **{predicted_digit}**")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            st.write("Prediction Probabilities:")
            st.bar_chart(prediction[0])

with tab2:
    st.header("Use a Random Test Sample")
    
    if st.button("🎲 Get Random Sample & Predict"):
        # Select random row
        random_index = np.random.randint(0, len(df))
        row = df.iloc[random_index]
        true_label = int(row.iloc[0])
        features = row.iloc[1:].values.astype('float32') / 255.0
        
        # Reshape to 28x28 for display
        img_display = features.reshape(28, 28)
        
        st.write(f"**True Label:** {true_label}")
        
        # Display image
        st.image(img_display, caption='Random Test Sample', width=150, clamp=True)
        
        # Predict
        prediction = model.predict(features.reshape(1, -1))
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        if predicted_digit == true_label:
            st.success(f"### 🎯 Predicted Digit: **{predicted_digit}** (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"### 🎯 Predicted Digit: **{predicted_digit}** (Confidence: {confidence:.2f}%)")
            
        st.write("Prediction Probabilities:")
        st.bar_chart(prediction[0])
