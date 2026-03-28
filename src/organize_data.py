import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Source of the dataset
dataset_path = 'visual-dataset'
# Target directory
data_dir = 'data'

# Class mapping (Source Folder -> Target Class)
class_map = {
    'Noload': 'Pass',
    'A10': 'Defect',
    'A&C10': 'Defect',
    'A&C&B10': 'Defect',
    'A30': 'Defect',
    'A&C30': 'Defect',
    'A&C&B30': 'Defect',
    'A50': 'Defect',
    'A&B50': 'Defect',
    'Fan': 'Defect',
    'Rotor-0': 'Defect'
}

# Create directory structure
for split in ['train', 'val']:
    for category in ['Pass', 'Defect']:
        os.makedirs(os.path.join(data_dir, split, category), exist_ok=True)

# Collect all images and their target classes
all_images = []
for folder, category in class_map.items():
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.bmp', '.png', '.jpg'))]
        for img in images:
            all_images.append((img, category))

# Split into train/validation (80/20) with stratification
train_imgs, val_imgs = train_test_split(
    all_images, test_size=0.2, random_state=42, stratify=[item[1] for item in all_images]
)

# Move files to new structure
def move_files(file_list, split):
    for src_path, category in file_list:
        filename = os.path.basename(src_path)
        # Using destination unique name by appending source folder name to prevent filename collision
        src_folder_name = os.path.basename(os.path.dirname(src_path))
        dest_filename = f"{src_folder_name}_{filename}"
        dest_path = os.path.join(data_dir, split, category, dest_filename)
        shutil.copy(src_path, dest_path)

print(f"Moving {len(train_imgs)} images to train/")
move_files(train_imgs, 'train')
print(f"Moving {len(val_imgs)} images to val/")
move_files(val_imgs, 'val')

print("Data organization complete.")
