# 🤟 Sign Language to Speech

A real-time system that converts **sign language gestures into speech** using computer vision and machine learning.  
This project bridges the communication gap between the hearing and speech-impaired communities by translating hand gestures into audible speech.

---

## 🚀 Features

- 🖐️ **Gesture Detection** – Recognizes sign language hand gestures via a webcam.  
- 🔊 **Speech Conversion** – Converts detected gestures into spoken words.  
- 📷 **Computer Vision** – Uses OpenCV for hand tracking and preprocessing.  
- 🤖 **Machine Learning** – Trained model for gesture classification.  
- 🎯 **User-Friendly** – Simple and accessible interface for real-time translation.  

---

## 📂 Project Structure

```
SignLanguage-to-Speech/
│── dataset/              # Dataset of hand gesture images
│── model/                # Trained ML models
│── notebooks/            # Jupyter notebooks for training and testing
│── src/                  # Source code for preprocessing, training, and inference
│── main.py               # Entry point to run the application
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Krisvarish/SignLanguage-to-Speech.git
   cd SignLanguage-to-Speech
   ```

2. **Create Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python main.py
   ```

---

## 🛠️ Technologies Used

- **Python 3**
- **OpenCV** – For image processing and hand tracking  
- **TensorFlow / Keras** – For training and running gesture recognition models  
- **Pyttsx3 / gTTS** – For converting text to speech  
- **Jupyter Notebook** – For experimentation and training  

---

## 📌 Future Improvements

- Expand dataset to cover more sign language gestures.  
- Improve real-time accuracy with advanced models (CNN, MediaPipe, etc.).  
- Add support for continuous sentence translation.  
- Build a mobile or web application for broader accessibility.  

---

## 🤝 Contributing

Pull requests are welcome! If you’d like to contribute new gestures, improve the model, or fix bugs, feel free to open an issue or PR.  
