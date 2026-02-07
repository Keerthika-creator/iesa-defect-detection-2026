# 📝 Code Examples - IESA DeepTech Hackathon 2026

**Practical examples for every use case**

---

## 🎯 Table of Contents

1. [Basic Training](#1-basic-training)
2. [Custom Training Loop](#2-custom-training-loop)
3. [Single Image Inference](#3-single-image-inference)
4. [Batch Inference](#4-batch-inference)
5. [Real-Time Camera Inference](#5-real-time-camera-inference)
6. [Model Export & Conversion](#6-model-export--conversion)
7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
8. [Custom Augmentation](#8-custom-augmentation)
9. [Error Analysis](#9-error-analysis)
10. [Production Deployment](#10-production-deployment)

---

## 1. Basic Training

### Minimal Example (Auto-pilot)

```python
from iesa_defect_detection_pipeline import *

# One-liner training
set_seed()
setup_directories()

# Auto-train both models
main()
```

### With Custom Dataset Path

```python
import os
from iesa_defect_detection_pipeline import *

# Override config
Config.DATASET_ROOT = "/path/to/your/Dataset"
Config.TRAIN_DIR = f"{Config.DATASET_ROOT}/train"
Config.VAL_DIR = f"{Config.DATASET_ROOT}/val"
Config.TEST_DIR = f"{Config.DATASET_ROOT}/test"

# Train
main()
```

---

## 2. Custom Training Loop

### Train Only Wafer Model

```python
from iesa_defect_detection_pipeline import *

# Setup
set_seed()
setup_directories()

# Build model
dual_model = DualHeadDefectModel(wafer_classes=8, die_classes=3)
wafer_model = dual_model.build_wafer_head()
dual_model.compile_model(wafer_model)

# Data generators
augmenter = IndustrialAugmentation()
train_gen = augmenter.get_train_generator(Config.TRAIN_DIR, Config.BATCH_SIZE)
val_gen = augmenter.get_val_generator(Config.VAL_DIR, Config.BATCH_SIZE)

# Compute class weights
sampler = AdaptiveSampler()
class_weights = sampler.compute_weights(train_gen)

# Train
trainer = TrainingPipeline(wafer_model, stage='wafer')
history = trainer.train(train_gen, val_gen, class_weights=class_weights)

# Visualize
trainer.plot_training_curves(save_path='my_training_curves.png')

# Save model
wafer_model.save('my_wafer_model.h5')
```

### Train with Custom Callbacks

```python
from tensorflow.keras.callbacks import LambdaCallback

# Custom callback: Print progress
def on_epoch_end(epoch, logs):
    print(f"Epoch {epoch+1}: val_acc={logs['val_accuracy']:.4f}")

custom_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# Add to training
trainer = TrainingPipeline(wafer_model, stage='wafer')
callbacks = trainer.get_callbacks()
callbacks.append(custom_callback)

history = wafer_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=Config.EPOCHS,
    callbacks=callbacks
)
```

---

## 3. Single Image Inference

### Basic Prediction

```python
from iesa_defect_detection_pipeline import InferencePipeline

# Load models
pipeline = InferencePipeline(
    wafer_model_path="outputs/models/wafer_best.h5",
    die_model_path="outputs/models/die_best.h5"
)

# Predict
result = pipeline.predict("test_image.jpg")

print(f"Stage: {result['stage']}")
print(f"Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Early Exit: {result['early_exit']}")
```

### With Tile Processing Disabled

```python
result = pipeline.predict(
    "small_image.jpg",
    use_tiling=False,  # Disable tiling
    early_exit=True
)
```

### Access All Class Probabilities

```python
result = pipeline.predict("test_image.jpg")

print("All Probabilities:")
for class_name, prob in result['all_probabilities'].items():
    print(f"  {class_name}: {prob:.2%}")
```

---

## 4. Batch Inference

### Process Multiple Images

```python
import glob
from tqdm import tqdm

# Get all test images
image_paths = glob.glob("Dataset/test/**/*.jpg", recursive=True)

# Initialize pipeline
pipeline = InferencePipeline(
    wafer_model_path="outputs/models/wafer_best.h5",
    die_model_path="outputs/models/die_best.h5"
)

# Batch predict
results = []
for img_path in tqdm(image_paths):
    result = pipeline.predict(img_path)
    results.append({
        'image': img_path,
        'class': result['predicted_class'],
        'confidence': result['confidence']
    })

# Save to CSV
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('batch_predictions.csv', index=False)
print(f"✓ Saved {len(results)} predictions")
```

### Parallel Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor

def predict_image(img_path):
    result = pipeline.predict(img_path)
    return {
        'image': img_path,
        'class': result['predicted_class'],
        'confidence': result['confidence']
    }

# Process in parallel (4 threads)
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(predict_image, image_paths))

print(f"Processed {len(results)} images in parallel")
```

---

## 5. Real-Time Camera Inference

### Webcam Stream

```python
import cv2
from iesa_defect_detection_pipeline import InferencePipeline
import numpy as np

# Load pipeline
pipeline = InferencePipeline(
    wafer_model_path="outputs/models/wafer_best.h5",
    die_model_path="outputs/models/die_best.h5"
)

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame temporarily
    cv2.imwrite('temp_frame.jpg', frame)
    
    # Predict
    result = pipeline.predict('temp_frame.jpg', use_tiling=False)
    
    # Draw results
    text = f"{result['predicted_class']} ({result['confidence']:.1%})"
    color = (0, 255, 0) if result['confidence'] > 0.8 else (0, 0, 255)
    
    cv2.putText(frame, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display
    cv2.imshow('Defect Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 6. Model Export & Conversion

### Export to ONNX

```python
from iesa_defect_detection_pipeline import ONNXExporter
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('outputs/models/wafer_best.h5')

# Export
exporter = ONNXExporter()
onnx_path = exporter.export_model(
    model=model,
    stage='wafer',
    opset=13,
    quantize=True
)

print(f"✓ ONNX model saved to {onnx_path}")
```

### Convert to TFLite (All Formats)

```python
from convert_to_tflite import TFLiteConverter

converter = TFLiteConverter(
    model_path='outputs/models/wafer_best.h5',
    output_path='wafer_int8.tflite'
)

# Convert to all formats
results = converter.convert_all()

print("\nGenerated Models:")
for format_name, path in results.items():
    print(f"  {format_name}: {path}")
```

### Convert Specific Format

```bash
# Command line
python convert_to_tflite.py \
  --model outputs/models/wafer_best.h5 \
  --output wafer_fp16.tflite \
  --format fp16
```

---

## 7. Hyperparameter Tuning

### Grid Search Over Learning Rates

```python
from iesa_defect_detection_pipeline import *

learning_rates = [1e-4, 1e-3, 1e-2]
results = {}

for lr in learning_rates:
    print(f"\nTesting LR={lr}")
    
    # Build model
    dual_model = DualHeadDefectModel()
    wafer_model = dual_model.build_wafer_head()
    dual_model.compile_model(wafer_model, learning_rate=lr)
    
    # Train
    trainer = TrainingPipeline(wafer_model, stage=f'wafer_lr{lr}')
    history = trainer.train(train_gen, val_gen)
    
    # Store results
    best_val_acc = max(history['val_accuracy'])
    results[lr] = best_val_acc
    
    print(f"Best Val Accuracy: {best_val_acc:.4f}")

# Find best LR
best_lr = max(results, key=results.get)
print(f"\n✓ Best LR: {best_lr} (Val Acc: {results[best_lr]:.4f})")
```

### Test Different Batch Sizes

```python
batch_sizes = [16, 32, 64]

for bs in batch_sizes:
    Config.BATCH_SIZE = bs
    
    # Recreate generators
    train_gen = augmenter.get_train_generator(Config.TRAIN_DIR, bs)
    val_gen = augmenter.get_val_generator(Config.VAL_DIR, bs)
    
    # Train
    # ... (same as above)
```

---

## 8. Custom Augmentation

### Add Custom Preprocessing

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def custom_preprocess(image):
    """Apply bilateral filtering for denoising"""
    import cv2
    return cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75)

# Create custom generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=5,
    preprocessing_function=custom_preprocess
)

train_gen = train_datagen.flow_from_directory(
    Config.TRAIN_DIR,
    target_size=Config.INPUT_SHAPE[:2],
    batch_size=Config.BATCH_SIZE,
    class_mode='categorical'
)
```

### Augmentation Pipeline with Albumentations

```bash
pip install albumentations
```

```python
import albumentations as A
import cv2
import numpy as np

# Define pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=5, p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.1, 
        contrast_limit=0.1, 
        p=0.5
    ),
    A.GaussNoise(var_limit=(10, 30), p=0.2)
])

def augment_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    augmented = transform(image=image)['image']
    
    return augmented / 255.0  # Normalize
```

---

## 9. Error Analysis

### Find Misclassified Images

```python
from sklearn.metrics import confusion_matrix
import numpy as np
import shutil

# Get predictions
test_gen = augmenter.get_val_generator(Config.TEST_DIR, Config.BATCH_SIZE)
model = tf.keras.models.load_model('outputs/models/wafer_best.h5')

predictions = model.predict(test_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes

# Find errors
errors = np.where(predicted_classes != true_classes)[0]

print(f"Found {len(errors)} misclassifications")

# Copy misclassified images
import os
os.makedirs('misclassified', exist_ok=True)

for idx in errors[:20]:  # First 20 errors
    img_path = test_gen.filepaths[idx]
    true_class = test_gen.class_indices_inv[true_classes[idx]]
    pred_class = test_gen.class_indices_inv[predicted_classes[idx]]
    
    # Copy with descriptive name
    new_name = f"true_{true_class}_pred_{pred_class}_{idx}.jpg"
    shutil.copy(img_path, f'misclassified/{new_name}')

print("✓ Misclassified images saved to misclassified/")
```

### Confidence Distribution by Class

```python
import matplotlib.pyplot as plt

# Get confidences per class
confidences_by_class = {}

for class_idx in range(len(Config.WAFER_CLASSES)):
    mask = (true_classes == class_idx)
    class_confidences = np.max(predictions[mask], axis=1)
    confidences_by_class[Config.WAFER_CLASSES[class_idx]] = class_confidences

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.boxplot(confidences_by_class.values(), labels=confidences_by_class.keys())
ax.set_ylabel('Confidence')
ax.set_title('Confidence Distribution by Class')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.savefig('confidence_by_class.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 10. Production Deployment

### Flask REST API

```python
from flask import Flask, request, jsonify
from iesa_defect_detection_pipeline import InferencePipeline
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load pipeline once
pipeline = InferencePipeline(
    wafer_model_path="outputs/models/wafer_best.h5",
    die_model_path="outputs/models/die_best.h5"
)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Predict
    result = pipeline.predict(filepath)
    
    # Clean up
    os.remove(filepath)
    
    return jsonify({
        'stage': result['stage'],
        'predicted_class': result['predicted_class'],
        'confidence': float(result['confidence']),
        'early_exit': result['early_exit']
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
```

**Test the API:**
```bash
curl -X POST -F "image=@test.jpg" http://localhost:5000/predict
```

### Docker Container

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY iesa_defect_detection_pipeline.py .
COPY outputs/models/ ./models/

# Expose port
EXPOSE 5000

# Run API
CMD ["python", "api.py"]
```

**Build & Run:**
```bash
docker build -t defect-detector .
docker run -p 5000:5000 defect-detector
```

### AWS Lambda Function

```python
import json
import base64
import boto3
from io import BytesIO
from PIL import Image
from iesa_defect_detection_pipeline import InferencePipeline

# Initialize pipeline (outside handler for reuse)
pipeline = InferencePipeline(
    wafer_model_path="/opt/models/wafer_best.h5",
    die_model_path="/opt/models/die_best.h5"
)

def lambda_handler(event, context):
    # Decode image from base64
    image_data = base64.b64decode(event['body'])
    image = Image.open(BytesIO(image_data))
    
    # Save temporarily
    image.save('/tmp/input.jpg')
    
    # Predict
    result = pipeline.predict('/tmp/input.jpg')
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'class': result['predicted_class'],
            'confidence': float(result['confidence']),
            'stage': result['stage']
        })
    }
```

---

## 🎯 Advanced Examples

### Ensemble Prediction

```python
# Load multiple checkpoints
models = [
    tf.keras.models.load_model('checkpoint_epoch20.h5'),
    tf.keras.models.load_model('checkpoint_epoch30.h5'),
    tf.keras.models.load_model('checkpoint_epoch40.h5')
]

def ensemble_predict(image_path):
    predictions = []
    
    for model in models:
        img = preprocess_image(image_path)
        pred = model.predict(img)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    
    return ensemble_pred

# Use ensemble
result = ensemble_predict('test.jpg')
```

### Test-Time Augmentation

```python
def predict_with_tta(model, image_path, num_augmentations=5):
    predictions = []
    
    for _ in range(num_augmentations):
        # Apply random augmentation
        augmented = augment_image(image_path)
        pred = model.predict(np.expand_dims(augmented, axis=0))
        predictions.append(pred)
    
    # Average
    return np.mean(predictions, axis=0)
```

---

**Need more examples? Check the full documentation in README.md**
