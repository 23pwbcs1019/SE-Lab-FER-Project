import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

def check_dataset():
    print(f"Checking data in: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print("ERROR: Data directory not found!")
        return

    emotions = os.listdir(DATA_DIR)
    total_images = 0
    valid_images = 0
    
    print(f"\nFound {len(emotions)} emotion classes: {emotions}")
    
    for emotion in emotions:
        folder_path = os.path.join(DATA_DIR, emotion)
        if not os.path.isdir(folder_path):
            continue
            
        images = os.listdir(folder_path)
        count = len(images)
        total_images += count
        
        print(f" -> {emotion}: {count} images")
        
        # Check first image dimensions
        if count > 0:
            img_path = os.path.join(folder_path, images[0])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != (48, 48):
                    print(f"    WARNING: Image size is {img.shape}, expected (48, 48)")
                else:
                    valid_images += count

    print("-" * 30)
    print(f"Total Images: {total_images}")
    print(f"Status: {'READY FOR TRAINING' if total_images > 0 else 'EMPTY - PLEASE CHECK FOLDERS'}")

if __name__ == "__main__":
    check_dataset()
    