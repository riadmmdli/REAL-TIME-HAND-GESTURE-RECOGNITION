import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Load trained model
model = load_model("C:/Users/riadm/Desktop/Real Time Sign Language Detection/sign_language_mobilenetv2_model.h5")

# Class labels
labels = ['Call me', 'Dislike', 'Hello', 'I Love You', 'Like', 'No', 'Okay', 'Peace', 'Thank you', 'Yes']

# Constants
offset = 20
imgSize = 224

# Webcam and detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # ✅ Allow up to 2 hands

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)

    if hands:
        for hand in hands:  # ✅ Loop through both hands
            x, y, w, h = hand['bbox']

            # Safe cropping
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(img.shape[1], x + w + offset)
            y2 = min(img.shape[0], y + h + offset)

            imgCrop = img[y1:y2, x1:x2]
            if imgCrop.size == 0:
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Prepare input
                imgInput = cv2.resize(imgWhite, (224, 224))
                imgInput = imgInput.astype("float32") / 255.0
                imgInput = np.expand_dims(imgInput, axis=0)

                # Predict
                prediction = model.predict(imgInput, verbose=0)
                index = np.argmax(prediction)
                confidence = np.max(prediction)

                label = f"{labels[index]} ({confidence*100:.1f}%)"
                cv2.putText(img, label, (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            except Exception as e:
                print("Prediction error:", e)
                continue

    # Show camera feed
    cv2.imshow("Two-Hand Sign Detection", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
