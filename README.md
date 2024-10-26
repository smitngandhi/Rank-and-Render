
# 🖼️ Blur or Clear Image Classification with NIMA Scoring & 🎬 Video Generation App

Welcome to the **Blur or Clear Image Classification** and **Video Generation** application! This app allows users to upload images to classify them as blurry or clear and get their NIMA scores. Additionally, users can generate videos by combining selected images with an audio track.

## 📦 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Functionality](#functionality)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Image Classification**: Classify uploaded images as blurry or clear using the model which is provided in another notebook. Run the notebook and save the model.
- **NIMA Scoring**: Evaluate the aesthetic quality of images with NIMA scores.
- **Video Generation**: Create a video from selected images with a background audio track.

## 🚀 Installation

To run this application locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/smitngandhi/Rank-and-Render.git
   cd Rank-and-Render
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your pre-trained model files (`blurOrClear.h5`, `mobilenet_weights.h5`) in the root directory.

4. Run the Streamlit app:
   ```bash
   streamlit run app.py --server.port 8051
   streamlit run video.py --server.port 8052
   ```

## 🖱️ Usage

1. **Image Classification**:
   - Upload one or more images in `.png`, `.jpg`, or `.jpeg` format.
   - Click the **Check and Generate Scores** button to classify the images.
   - The results will display whether the images are clear or blurry, along with their NIMA scores.

2. **Video Generation**:
   - Select the number of images to include in the video.
   - Upload an audio file in `.mp3` format.
   - Click the button to generate the video.

## ⚙️ Functionality

### Image Classification & NIMA Scoring

- The app uses a pre-trained model to classify images as either **blurry** or **clear**.
- NIMA (Neural Image Assessment) scores are computed to evaluate the aesthetic quality of the images.

### Video Generation

- Users can create a video by combining selected images and an audio track.
- The app generates a video file, which can be downloaded directly.

## 💡 How It Works

1. **Image Processing**:
   - Uploaded images are resized and converted to grayscale for classification.
   - Predictions are made using a TensorFlow model, and the classification results are displayed.

2. **NIMA Model**:
   - A MobileNet model is used to calculate NIMA scores for the classified images.
   - The scores are calculated and displayed, along with standard deviation values.

3. **Video Creation**:
   - Selected images are combined to create a video, with the audio file synchronized to the video length.
   - The video is saved and can be downloaded by the user.

## 🤝 Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request.

Please make sure to follow the standard guidelines for contributing to open-source projects.

Thank you for using the Blur or Clear Image Classification with NIMA Scoring and Video Generation App! 🎉

For any questions or feedback, feel free to reach out.
