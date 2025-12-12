import os
import shutil

# Path to your data
DATA_DIR = os.path.join('data', 'raw')

def clean_ghosts():
    print(f"🧹 Scanning {DATA_DIR} for junk files...")
    if not os.path.exists(DATA_DIR):
        print("❌ Error: Data folder not found.")
        return

    # Get list of emotion folders
    if os.path.exists(os.path.join(DATA_DIR, 'CK+48')): 
        print("⚠️ Found CK+48 subfolder. Please move folders up to data/raw/ directly.")
        return

    items = os.listdir(DATA_DIR)
    clean_count = 0
    valid_folders = []

    for item in items:
        item_path = os.path.join(DATA_DIR, item)
        
        # 1. Delete loose files (like .DS_Store or desktop.ini)
        if os.path.isfile(item_path):
            print(f"   ❌ Removing file: {item}")
            os.remove(item_path)
            clean_count += 1
            
        # 2. Delete hidden folders (like .ipynb_checkpoints)
        elif item.startswith('.'):
            print(f"   ❌ Removing hidden folder: {item}")
            shutil.rmtree(item_path)
            clean_count += 1
        
        # 3. Keep valid folders
        elif os.path.isdir(item_path):
            valid_folders.append(item)

    print(f"\n✅ Cleanup Complete. Removed {clean_count} junk items.")
    print(f"📂 Valid Emotion Folders ({len(valid_folders)}):")
    print(sorted(valid_folders))

if __name__ == "__main__":
    clean_ghosts()