import streamlit as st
from PIL import Image
from models.model_loader import load_model
from utils.image_processing import preprocess_image
from utils.prediction import predict
from utils.descriptions import classes, descriptions, gejala, pencegahan, penanganan
import os
import numpy as np

# Ignore warnings
os.environ["PYTHONWARNINGS"] = "ignore"

# Print numpy version for debugging
print(f"Numpy Version: {np.__version__}")

# Set custom CSS
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .description .gejala .pencegahan .penanganan {
            font-size: 18px;
            text-align: center;
            color: #555555;
        }
        .prediction-container {
            background-color: #f4f4f9;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .result-text {
            font-size: 20px;
            font-weight: bold;
            color: #4CAF50;
        }
        .confidence {
            font-size: 18px;
            color: #FF9800;
        }
        .upload-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            width: 200px;
        }
        .upload-btn:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Main function for Streamlit app
def main():
    st.markdown('<div class="title">Dragonfruit Stem Health Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Upload a dragonfruit stem image to detect its health condition.</div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    model_path = "resnet50_dragonfruit.pth"

    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Add a "Deteksi" button
        if st.button("Deteksi"):
            # Load model
            model = load_model(model_path)

            # Preprocess image
            preprocessed_image = preprocess_image(image)

            # Make prediction
            class_name, confidence = predict(preprocessed_image, model, classes)

            # Display results in styled container
            st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-text">Prediction: {class_name}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="description">{descriptions[class_name]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="gejala">{gejala[class_name]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="pencegahan">{pencegahan[class_name]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="penanganan">{penanganan[class_name]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
