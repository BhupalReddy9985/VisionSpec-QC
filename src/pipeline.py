import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy as np

# Config
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

def get_data_generators(data_dir='data'):
    """Creates ImageDataGenerators for training and validation."""
    
    # Advanced Data Augmentation for Training
    train_datagen = ImageDataGenerator(
        rescale=1./255,               # Pixel Normalization (0-1)
        rotation_range=20,            # Robust to motor/camera rotation
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,               # Robust to varying distances
        horizontal_flip=True,         # Flip invariance
        brightness_range=[0.7, 1.3],  # Robust to lighting changes
        fill_mode='nearest'
    )

    # Simple Rescaling for Validation
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',          # Pass vs Defect
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator

def visualize_augmented_samples(train_gen, n_samples=5):
    """Saves a plot of original/augmented samples to verify realism."""
    plt.figure(figsize=(15, 6))
    
    # Get a single batch
    images, labels = next(train_gen)
    
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.imshow(images[i])
        class_name = 'Defect' if labels[i] == 0 else 'Pass' # Check directory structure to confirm mapping
        # Label check: Pass is likely 1, Defect is 0 (alphabetic order: D, P)
        plt.title(f"Label: {class_name}")
        plt.axis('off')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/augmented_samples.png')
    print("Augmented samples saved to results/augmented_samples.png")

if __name__ == "__main__":
    train_gen, val_gen = get_data_generators()
    visualize_augmented_samples(train_gen)
    print("Class indices:", train_gen.class_indices)
