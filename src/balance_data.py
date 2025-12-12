import os
import shutil

DATA_DIR = os.path.join('data', 'raw')
TARGET_COUNT = 150  # Force small classes to have at least this many images

def balance_folders():
    if not os.path.exists(DATA_DIR):
        print("Error: Data folder not found")
        return

    folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')]
    
    for emotion in folders:
        folder_path = os.path.join(DATA_DIR, emotion)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(images)
        
        print(f"Checking {emotion}: {current_count} images...")
        
        if current_count < TARGET_COUNT and current_count > 0:
            print(f"   ⚠️ Boosting {emotion} from {current_count} to {TARGET_COUNT}...")
            
            # Duplicate existing images until we hit the target
            needed = TARGET_COUNT - current_count
            for i in range(needed):
                src_img = images[i % current_count] # Cycle through existing images
                src_path = os.path.join(folder_path, src_img)
                
                # Create a copy with a unique name
                dst_name = f"aug_{i}_{src_img}"
                dst_path = os.path.join(folder_path, dst_name)
                shutil.copy(src_path, dst_path)
            
            print(f"   ✅ Done. Now has {len(os.listdir(folder_path))} images.")

if __name__ == "__main__":
    balance_folders()