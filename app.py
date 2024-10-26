import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.layers import Dense, Dropout
from PIL import Image
import os
from pathlib import Path
import csv
import shutil
import tensorflow as tf

# Load the pre-trained model for blur or clear classification
model = load_model('blurOrClear.h5')

# Add custom CSS styling
st.markdown("""
    <style>
    /* Add your custom CSS styling here */
    body {
        background-color: #f0f2f5;
    }
    .stImage {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Create the Streamlit app
st.title("Blur or Clear Image Classification with NIMA Scoring")
st.write("Upload one or more images to check if they're blurry or clear and get their NIMA scores.")

# Set image upload format and conditions
uploaded_files = st.file_uploader("Choose one or more images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Add a button to trigger the processing of uploaded images
if st.button("Check and Generate Scores"):
    if os.path.exists('sorted_images'):
        shutil.rmtree('sorted_images')
    # Define the minimum dimensions (512x512 pixels)
    min_width = 512
    min_height = 512

    # Path to save images
    save_dir = "saved_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Process each uploaded file
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Processing: **{uploaded_file.name}**")

            # Open the uploaded image using PIL
            image = Image.open(uploaded_file)
            original_image = np.array(image)  # Keep the original image for later saving

            # Check image dimensions
            if image.width < min_width or image.height < min_height:
                st.error(f"Image {uploaded_file.name} must be at least {min_width}x{min_height} pixels.")
                continue

            # Display the uploaded image
            st.image(image, caption=f"Uploaded Image - {uploaded_file.name}", use_column_width=True)

            # Convert image to grayscale and process it for classification
            gray_image = image.convert('L')  # Convert to grayscale
            gray_image = np.array(gray_image)
            gray_image = cv2.resize(gray_image, (512, 512))  # Resize to 512x512
            gray_image = gray_image.reshape((1, 512, 512, 1))  # Reshape for the model
            gray_image = gray_image.astype('float32') / 255  # Normalize to [0, 1]

            # Make prediction
            prediction = model.predict(gray_image)

            # Check the prediction
            if prediction > 0.5:
                st.success(f"The image {uploaded_file.name} is classified as **clear**.")
                # Save the original color image
                save_path = os.path.join(save_dir, uploaded_file.name)
                cv2.imwrite(save_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
                st.write(f"Original color image saved at: {save_path}")
            else:
                st.warning(f"The image {uploaded_file.name} is classified as **blurry**.")
                

        # NIMA Scoring

        # Define the Args class to replace command-line arguments
        class Args:
            def __init__(self):
                self.dir = save_dir  # Directory where saved images are
                self.img = None
                self.resize = 'true'
                self.rank = 'true'

        args = Args()

        # Define utility functions
        def mean_score(scores):
            si = np.arange(1, 11, 1)
            mean = np.sum(scores * si)
            return mean

        def std_score(scores):
            si = np.arange(1, 11, 1)
            mean = mean_score(scores)
            std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
            return std

        # Process arguments
        resize_image = args.resize.lower() in ("true", "yes", "t", "1")
        target_size = (224, 224) if resize_image else None
        rank_images = args.rank.lower() in ("true", "yes", "t", "1")

        # Load images
        st.write("Loading images from directory:", args.dir)
        imgs = [str(f) for f in Path(args.dir).glob('*.png')]
        imgs += [str(f) for f in Path(args.dir).glob('*.jpg')]
        imgs += [str(f) for f in Path(args.dir).glob('*.jpeg')]

        # Set up the NIMA model
        with tf.device('/CPU:0'):
            base_model = MobileNet((224, 224, 3), alpha=1, include_top=False, pooling='avg', weights=None)
            x = Dropout(0.75)(base_model.output)
            x = Dense(10, activation='softmax')(x)
            nima_model = Model(base_model.input, x)

            # Load NIMA model weights
            nima_model.load_weights('mobilenet_weights.h5')  # Update with the correct path to weights

            score_list = []
            for img_path in imgs:
                # Open the image using PIL
                img = Image.open(img_path)
                if target_size:
                    img = img.resize(target_size)

                # Convert PIL Image to numpy array
                x = np.array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                scores = nima_model.predict(x, batch_size=1, verbose=0)[0]
                mean = mean_score(scores)
                std = std_score(scores)

                file_name = Path(img_path).name.lower()
                score_list.append((file_name, mean))

                st.write("Evaluating:", img_path)
                st.write(f"NIMA Score: {mean:.3f} ± ({std:.3f})")

            if rank_images:
                st.write("*" * 40, "Ranking Images", "*" * 40)
                score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
                for i, (name, score) in enumerate(score_list):
                    st.write(f"{i + 1}) {name} : Score = {score:.5f}")

            # Save results to a CSV file
            with open('nima_scores.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Image", "NIMA Score"])
                for name, score in score_list:
                    writer.writerow([name, score])

            st.write("Scores have been saved to 'nima_scores.csv'.")

        # Create a directory to store sorted images
        sorted_dir = "sorted_images"
        if not os.path.exists(sorted_dir):
            os.makedirs(sorted_dir)

        # Move images to the sorted folder based on scores
        for i, (name, score) in enumerate(score_list):
            src_path = os.path.join(save_dir, name)
            dst_path = os.path.join(sorted_dir, f"{i+1}_{name}")
            shutil.move(src_path, dst_path)

        st.write(f"Images sorted and saved in the '{sorted_dir}' folder.")
        shutil.rmtree('saved_images')


# Redirect to another app
# if st.button('Go to Video Generation App'):
#     # JavaScript for redirection
#     js = "window.open('http://localhost:8502', '_self');"  # Opens in the same tab
#     st.markdown(f"<script>{js}</script>", unsafe_allow_html=True)
