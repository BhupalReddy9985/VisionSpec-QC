import time
import tensorflow as tf
import os
import cv2
import numpy as np
import random
from explainability import get_gradcam_heatmap, overlay_heatmap
import matplotlib.pyplot as plt

# Config
MODEL_PATH = 'model_output/visionspec_qc_v1.h5'
IMG_SIZE = (224, 224)
SAMPLE_DIR = 'data/val/Defect' # Test on defects to see the Grad-CAM in action

def run_live_simulation(n_frames=5):
    """Simulates processing a live video stream frame-by-frame with latency tracking."""
    print("--- VisionSpec QC: Live Inference Simulation ---")
    
    # Load Model
    start_load = time.time()
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded in {time.time() - start_load:.2f} seconds.")
    
    # Get random sample images
    pass_imgs = [os.path.join('data/val/Pass', f) for f in os.listdir('data/val/Pass')]
    defect_imgs = [os.path.join('data/val/Defect', f) for f in os.listdir('data/val/Defect')]
    
    test_samples = pass_imgs + defect_imgs
    random.shuffle(test_samples)
    
    for i, img_path in enumerate(test_samples[:n_frames]):
        print(f"\n[Frame {i+1}] Processing {os.path.basename(img_path)}...")
        
        # Inference & Latency
        start_inf = time.time()
        
        # Preprocess
        raw_img = cv2.imread(img_path)
        img = cv2.resize(raw_img, IMG_SIZE)
        img_array = np.expand_dims(img, axis=0) / 255.0
        
        # Predict
        preds = model.predict(img_array, verbose=0)
        prediction = preds[0][0]
        label = "PASS" if prediction > 0.5 else "DEFECT"
        confidence = prediction if prediction > 0.5 else (1 - prediction)
        
        inf_latency = (time.time() - start_inf) * 1000 # in ms
        print(f"Latency: {inf_latency:.2f} ms | Result: {label} (Conf: {confidence:.2f})")
        
        # Grad-CAM if Defect detected or for inspection
        heatmap = get_gradcam_heatmap(img_array, model, 'out_relu')
        overlay = overlay_heatmap(img_path, heatmap)
        
        # Show comparison
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original: {os.path.basename(img_path)}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        status_color = 'green' if label == 'PASS' else 'red'
        plt.title(f"Grad-CAM Result: {label}", color=status_color)
        plt.axis('off')
        
        os.makedirs('results/demo', exist_ok=True)
        plt.savefig(f'results/demo/inference_frame_{i+1}.png')
        plt.close()

    print("\n--- Simulation Complete. Frames saved to results/demo/ ---")

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        run_live_simulation()
    else:
        print("Model file not found. Ensure training is complete.")
