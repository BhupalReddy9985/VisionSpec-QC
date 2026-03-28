import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import layers

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for the given image and model.
    """
    # Access the base model (first layer of the sequential model)
    base_model = model.layers[0]
    
    # Create a model for the classification head (layers after the base model)
    # The classification head starts from layer 1 onwards
    head_input = layers.Input(shape=base_model.output.shape[1:])
    x = head_input
    for layer in model.layers[1:]:
        x = layer(x)
    head_model = Model(head_input, x)

    # Create a model for the base model's internal layer
    grad_model = Model(
        inputs=[base_model.inputs],
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, base_preds = grad_model(img_array)
        # Pass base model output through the head model
        preds = head_model(base_preds)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the class output wrt the feature maps
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Global Average Pooling of the gradients (weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps by the pooled gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM heatmap onto the original image.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Rescale heatmap (0-255) and resize to original image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Superimpose heatmap
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

def visualize_defect_location(model_path, img_path, save_path):
    """
    Comprehensive utility for Grad-CAM visualization.
    Target layer for MobileNetV2 is often 'out_relu' (the final 7x7 conv).
    """
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0) / 255.0
    
    # Generate Heatmap
    # Note: For MobileNetV2, 'out_relu' is a good target layer
    try:
        heatmap = get_gradcam_heatmap(img_array, model, 'out_relu')
        result_img = overlay_heatmap(img_path, heatmap)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(result_img)
        plt.title("VisionSpec Heatmap (Defect Location)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Explainability report saved to {save_path}")
    except Exception as e:
        print(f"Error during Grad-CAM generation: {e}")

if __name__ == "__main__":
    # Test on a known defect image
    # Note: we need the model first. 
    # This script will be called in Week 3 demonstration logic.
    pass
