import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- CONFIGURATION ---
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 60
DATA_DIR = os.path.join('data', 'raw')
MODEL_PATH = os.path.join('models', 'emotion_model.h5')

def load_data():
    print("Loading Data...")
    images = []
    labels = []
    # Load all folders (Anger, Disgust, Fear, Happy, Neutral, Sadness, Surprise)
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')])
    class_map = {name: i for i, name in enumerate(class_names)}
    
    for emotion in class_names:
        folder = os.path.join(DATA_DIR, emotion)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(class_map[emotion])
                
    X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    y = to_categorical(np.array(labels))
    return X, y, len(class_names)

def build_model(num_classes):
    model = Sequential([
        # Standard CNN Architecture (Proven Stability)
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y, num_classes = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Standard Augmentation
    datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.15, horizontal_flip=True)
    
    model = build_model(num_classes)
    print("Starting Training (Restoring Accuracy)...")
    model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data=(X_test, y_test), epochs=EPOCHS)
    
    if not os.path.exists('models'): os.makedirs('models')
    model.save(MODEL_PATH)
    print("SUCCESS: Accurate Model Saved.")