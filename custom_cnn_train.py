import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ---------------------- CONFIG ----------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
MODEL_NAME = "Custom CNN"
DATA_DIR = "C:/Users/riadm/Desktop/Real Time Sign Language Detection/DataSet"
MODEL_PATH = "custom_cnn_model.h5"

# ---------------------- DATA ----------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ---------------------- MODEL ----------------------
def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

model = build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=train_gen.num_classes)

# ---------------------- COMPILE & TRAIN ----------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1, min_lr=1e-6)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# ---------------------- SAVE MODEL ----------------------
model.save(MODEL_PATH)

# ---------------------- PLOT TRAINING CURVES ----------------------
plt.figure(figsize=(12, 5))
plt.suptitle(f"{MODEL_NAME} Training Curves", fontsize=16, fontweight='bold')

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---------------------- EVALUATION ----------------------
val_gen.reset()
Y_true = val_gen.classes
Y_pred = model.predict(val_gen, verbose=1)
Y_pred_classes = np.argmax(Y_pred, axis=1)
labels = list(val_gen.class_indices.keys())

# General Metrics
acc = accuracy_score(Y_true, Y_pred_classes)
precision = precision_score(Y_true, Y_pred_classes, average='weighted', zero_division=0)
recall = recall_score(Y_true, Y_pred_classes, average='weighted', zero_division=0)
f1 = f1_score(Y_true, Y_pred_classes, average='weighted', zero_division=0)

print("\n🎯 Model Evaluation:")
print(f"✔️ Accuracy : {acc*100:.2f}%")
print(f"✔️ Precision: {precision*100:.2f}%")
print(f"✔️ Recall   : {recall*100:.2f}%")
print(f"✔️ F1-Score : {f1*100:.2f}%")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(Y_true, Y_pred_classes, target_names=labels, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f"{MODEL_NAME} - Confusion Matrix", fontsize=16, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ---------------------- OPTIONAL: SAMPLE PREDICTIONS ----------------------
print("\n🔍 Sample Predictions:")
for i in range(10):
    print(f"True: {labels[Y_true[i]]} | Predicted: {labels[Y_pred_classes[i]]}")
