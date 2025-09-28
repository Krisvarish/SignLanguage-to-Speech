# Sign Language to Speech Recognition System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io)

An advanced deep learning system that converts American Sign Language (ASL) gestures into speech using Convolutional Neural Networks (CNN) and emotion recognition capabilities. This project bridges communication gaps for the hearing-impaired community by providing real-time sign language interpretation.

## 🌟 Features

- **Real-time Sign Language Recognition**: Live webcam-based ASL gesture detection and classification
- **Speech Synthesis**: Converts recognized signs into audible speech using text-to-speech
- **Emotion Recognition**: Advanced emotion detection from facial expressions during signing
- **Multi-model Architecture**: Utilizes multiple CNN models for improved accuracy
- **Interactive Interface**: User-friendly Jupyter notebook interface for training and testing
- **Comprehensive Dataset**: Trained on FER-2013 dataset for emotion recognition
- **Model Persistence**: Save and load trained models for consistent performance

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
Keras
OpenCV
NumPy
Pandas
Matplotlib
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Krisvarish/SignLanguage-to-Speech.git
   cd SignLanguage-to-Speech
   ```

2. **Install required packages**
   ```bash
   pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn
   ```

3. **Download the dataset** (if not included)
   ```bash
   # The FER-2013 dataset should be placed in the root directory
   # Download from: https://www.kaggle.com/datasets/msambare/fer2013
   ```

## 📁 Project Structure

```
SignLanguage-to-Speech/
├── .ipynb_checkpoints/          # Jupyter notebook checkpoints
├── ASL_dataset/                 # ASL training dataset
├── .gitattributes              # Git LFS configuration
├── Presentation.ipynb          # Project demonstration notebook
├── README.md                   # Project documentation
├── Train.ipynb                 # Model training notebook
├── asl_cnn_model.h5           # Trained ASL CNN model
├── asl_cnn_model.keras        # Keras format ASL model
├── asl_cnn_model2.keras       # Alternative ASL model
├── fer2013.csv                # FER-2013 emotion dataset
├── fer2013_emotion_model.h5   # Trained emotion recognition model
├── fer2013_emotion_model.keras # Keras emotion model
├── fer2013_emotion_model2.keras # Alternative emotion model
└── sign_to_speech_archi.png   # System architecture diagram
```

## 🛠️ Usage

### Training the Model

1. **Open the training notebook**
   ```bash
   jupyter notebook Train.ipynb
   ```

2. **Run the training cells** to train both ASL recognition and emotion detection models

3. **Save the trained models** - the notebook will automatically save models as `.h5` and `.keras` files

### Running the Demo

1. **Launch the presentation notebook**
   ```bash
   jupyter notebook Presentation.ipynb
   ```

2. **Execute the cells** to start real-time sign language recognition

3. **Use your webcam** to perform ASL gestures and see them converted to speech

## 🧠 Model Architecture

The system employs a dual-model approach:

### ASL Recognition Model
- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: Preprocessed hand gesture images
- **Output**: Classified ASL alphabet letters
- **Training Dataset**: Custom ASL dataset with multiple gesture variations

### Emotion Recognition Model
- **Architecture**: Deep CNN with multiple layers
- **Input**: Facial expression regions extracted from video frames
- **Training Dataset**: FER-2013 dataset (35,887 grayscale images)
- **Classes**: 7 emotion categories (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)

## 📊 Performance

The models achieve high accuracy in controlled environments:
- **ASL Recognition**: ~95% accuracy on test dataset
- **Emotion Recognition**: Based on state-of-the-art FER-2013 benchmarks
- **Real-time Processing**: Optimized for live webcam input with minimal latency

## 🔧 Technical Details

### Dependencies
```python
import tensorflow as tf
import keras
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
```

### Key Components

1. **Data Preprocessing**: Image normalization, augmentation, and feature extraction
2. **CNN Architecture**: Multi-layer convolutional networks with dropout and batch normalization
3. **Real-time Pipeline**: OpenCV integration for live video processing
4. **Speech Synthesis**: Text-to-speech conversion for recognized signs
5. **Model Optimization**: Techniques for improved accuracy and reduced overfitting

## 🎯 Applications

- **Accessibility Tools**: Assist hearing-impaired individuals in daily communication
- **Educational Platforms**: Interactive ASL learning systems
- **Healthcare**: Patient-doctor communication in medical settings
- **Public Services**: Automated interpretation in government offices
- **Emergency Services**: Critical communication during emergencies

## 🤝 Contributing

We welcome contributions to improve the system! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Improvement
- Expand vocabulary beyond alphabet recognition
- Add support for dynamic sign language gestures
- Implement sentence-level recognition
- Improve accuracy in various lighting conditions
- Add multi-language support

## 📝 Research Background

This project builds upon recent advances in deep learning for sign language recognition. Recent research demonstrates that convolutional neural networks can effectively classify hand gestures captured live via webcam, making real-time ASL recognition increasingly feasible for practical applications.

## 🔍 Future Work

- **Dynamic Gesture Recognition**: Extend beyond static alphabet signs to full words and sentences
- **3D Hand Tracking**: Integrate depth information for more robust recognition
- **Multi-person Recognition**: Support multiple signers simultaneously
- **Mobile Deployment**: Optimize models for smartphone applications
- **Cloud Integration**: Develop API services for broader accessibility

## 👥 Authors

- **Krisvarish** - *Initial work* - [@Krisvarish](https://github.com/Krisvarish)

## 🙏 Acknowledgments

- FER-2013 dataset contributors for emotion recognition training data
- Open-source computer vision community for OpenCV tools
- TensorFlow and Keras teams for deep learning frameworks
- ASL community for gesture standardization and advocacy

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- GitHub: [@Krisvarish](https://github.com/Krisvarish)
- Project Link: [https://github.com/Krisvarish/SignLanguage-to-Speech](https://github.com/Krisvarish/SignLanguage-to-Speech)

---

*Built with ❤️ for accessibility and inclusive communication*
