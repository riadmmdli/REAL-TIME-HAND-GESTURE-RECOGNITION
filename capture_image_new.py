import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import random

# Webcam and detectors
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224

# Dataset folder & labels
folderPath = "C:/Users/riadm/Desktop/Real Time Sign Language Detection/DataSet/"
keywords = ["Call me", "Dislike", "Hello", "I Love You", "Like", "No", "Okay", "Peace", "Thank you", "Yes"]
currentIndex = 0
counter = 0

# Augmentation functions
def random_brightness_contrast(image):
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.randint(-30, 30)    # brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def maybe_flip(image):
    if random.random() < 0.5:
        return cv2.flip(image, 1)
    return image

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    imgWithLabel = img.copy()

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
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

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        except:
            print("Crop out of bounds. Move hand to center.")

    # Show label
    cv2.putText(imgWithLabel, f"{keywords[currentIndex]} ({counter})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Webcam", imgWithLabel)

    key = cv2.waitKey(1)
    
    if key == ord("s") and hands:
        # Apply augmentation before saving
        aug_image = maybe_flip(imgWhite)
        aug_image = random_brightness_contrast(aug_image)

        save_dir = os.path.join(folderPath, keywords[currentIndex])
        os.makedirs(save_dir, exist_ok=True)

        filename = f'{save_dir}/Image_{time.time()}.jpg'
        cv2.imwrite(filename, aug_image)
        counter += 1
        print(f"[{keywords[currentIndex]}] Saved image {counter}")

    elif key == ord("u"):
        currentIndex = min(currentIndex + 1, len(keywords) - 1)
        counter = 0
    elif key == ord("y"):
        currentIndex = max(currentIndex - 1, 0)
        counter = 0
    elif key == ord("q"):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
