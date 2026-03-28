import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import os
from pipeline import get_data_generators

# Config
IMG_SHAPE = (224, 224, 3)
EPOCHS = 10  # Reduced for demo performance; Increase in production
LR = 1e-4

def build_model():
    """Builds the Transfer Learning model using MobileNetV2."""
    # Base model: MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze early layers
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),          # Prevent overfitting
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Binary classification
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_vision_spec():
    """Trains the VisionSpec QC model with Class Weights."""
    train_gen, val_gen = get_data_generators()
    
    # Calculate Class Weights (Imbalance Handling)
    # Total Pass: 25, Total Defect: 344
    n_pass = 25
    n_defect = 344
    n_total = n_pass + n_defect
    
    # Weight = Total / (n_classes * n_samples)
    weight_for_defect = (n_total) / (2 * n_defect)
    weight_for_pass = (n_total) / (2 * n_pass)
    
    class_weights = {0: weight_for_defect, 1: weight_for_pass}
    print(f"Applying Class Weights: {class_weights}")
    
    model = build_model()
    
    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model_output/visionspec_qc_v1.h5', save_best_only=True)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stop, checkpoint]
    )
    
    # Save training curves
    plot_learning_curves(history)
    print("Model training complete. Weights saved to model_output/visionspec_qc_v1.h5")
    return history

def plot_learning_curves(history):
    """Plots Training vs Validation curves for analysis."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/learning_curves.png')
    print("Learning curves saved to results/learning_curves.png")

if __name__ == "__main__":
    train_vision_spec()
