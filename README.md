# ✋ Real-Time Hand Gesture Recognition using CNN & MobileNetV2

This project demonstrates a real-time hand gesture recognition system using computer vision and deep learning. The system uses a webcam to detect hand signs and classify them into predefined categories using either a **custom-trained CNN** or a **MobileNetV2 transfer learning model**.  

🎥 **Demo Video**: 

https://github.com/user-attachments/assets/573cbd8c-2ece-47f7-a10c-e1e83618a556



---

## 📌 Features

- 🔍 Real-time hand detection using **MediaPipe (cvzone)**  
- 🧠 Gesture classification using:
  - Custom CNN model
  - MobileNetV2 (pretrained on ImageNet + fine-tuned)
- 🖼 Image preprocessing: cropping, resizing, centering on white background
- 🏷 Prediction displayed on webcam feed with class label and confidence
- 🎓 Achieved:
  - CNN Test Accuracy: ~78%
  - MobileNetV2 Test Accuracy: ~91%

---

## 📁 Dataset

This project uses a **custom hand gesture dataset** generated via webcam, consisting of 10 gesture classes:

- Call me, Dislike, Hello, I Love You, Like, No, Okay, Peace, Thank you, Yes

**📌 Note**: If you would like to use this dataset for academic or personal purposes, please contact the author for **permission first**.

---

## 🧠 Models

### ✅ Custom CNN
- Built from scratch using Keras
- 4 convolutional blocks with dropout and dense layers
- Trained for 50 epochs
- Validation accuracy: ~80%

### ✅ MobileNetV2
- Transfer learning from ImageNet
- Only top layers trained
- Validation accuracy: ~93%
- Generalizes better and faster

---

## 🛠 Installation

```bash
git clone https://github.com/riadmmdli/REAL-TIME-HAND-GESTURE-RECOGNITION.git
cd REAL-TIME-HAND-GESTURE-RECOGNITION
pip install -r requirements.txt
```
## 🚀 How to Run
### 📸 1. Capture Dataset
bash
Copy
Edit
python capture_image_new.py
Use keys u/y to switch between classes and s to save images.

### 🧠 2. Train Model
bash
Copy
Edit
python custom_cnn_train.py        # Train custom CNN
python mobilenetv2_train.py      # Train MobileNetV2
### 📈 3. Evaluate Model
bash
Copy
Edit
python test_custom_cnn.py        # Test CNN
python mobilenetv2_test.py       # Test MobileNetV2

### 🤖 4. Run Real-Time Prediction
bash
Copy
Edit
python realtime_prediction.py    # or your own version

### 🔧 Requirements
Python 3.7+
TensorFlow
OpenCV
NumPy
cvzone (uses MediaPipe under the hood)
matplotlib, seaborn (for plotting and evaluation)

### 📊 Performance Summary
Model	Train Acc	Val Acc	Test Acc
Custom CNN	95%	80%	~78%
MobileNetV2	97%	93%	~91%

## 🧠 Acknowledgments
TensorFlow/Keras
cvzone
MediaPipe
YouTube tutorials and open-source contributors for gesture datasets

## 📄 License
This project is licensed under the MIT License.










