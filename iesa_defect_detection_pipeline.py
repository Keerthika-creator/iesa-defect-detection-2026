"""
IESA DeepTech Hackathon 2026 - Dual-Stage Semiconductor Defect Detection System
================================================================================

Edge AI solution with stage-aware inference, tile-based processing, and 
lightweight MobileNetV2 architecture for wafer and die defect classification.

Author: Claude AI Engineering Team
Target Platform: NXP eIQ Edge Devices
Training Environment: Google Colab
"""

import os
import sys
import json
import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)

# Image Processing
import cv2
from PIL import Image

# ML Utilities
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight

# Model Export
import onnx
import tf2onnx

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

print("=" * 80)
print("IESA DeepTech Hackathon 2026 - Edge AI Defect Detection System")
print("=" * 80)
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("=" * 80)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for the defect detection pipeline"""
    
    # Dataset Paths
    DATASET_ROOT = "/content/Dataset"
    TRAIN_DIR = f"{DATASET_ROOT}/train"
    VAL_DIR = f"{DATASET_ROOT}/val"
    TEST_DIR = f"{DATASET_ROOT}/test"
    
    # Output Paths
    OUTPUT_DIR = "/content/outputs"
    MODEL_DIR = f"{OUTPUT_DIR}/models"
    RESULTS_DIR = f"{OUTPUT_DIR}/results"
    LOGS_DIR = f"{OUTPUT_DIR}/logs"
    
    # Model Architecture
    INPUT_SHAPE = (224, 224, 3)
    TILE_SIZE = 224
    TILE_OVERLAP = 32
    BACKBONE = "MobileNetV2"
    ALPHA = 0.75  # MobileNet width multiplier for edge optimization
    
    # Class Definitions
    WAFER_CLASSES = [
        "Clean", "Center", "Donut", "Edge-Loc", 
        "Edge-Ring", "Loc", "Scratch", "Other"
    ]
    DIE_CLASSES = ["Good", "Defective", "Unknown"]
    
    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 50
    INITIAL_LR = 1e-3
    FINE_TUNE_LR = 1e-4
    FINE_TUNE_EPOCH = 20  # Unfreeze backbone after this epoch
    
    # Stage Detection Thresholds
    WAFER_SIZE_THRESHOLD = 800  # pixels (min dimension)
    REPETITION_THRESHOLD = 0.7  # pattern repetition score
    EDGE_DENSITY_THRESHOLD = 0.15  # edge pixel ratio
    
    # Confidence Thresholds
    CONFIDENCE_THRESHOLD = 0.6
    EARLY_EXIT_CONFIDENCE = 0.95
    
    # Edge Optimization
    QUANTIZATION = True
    ONNX_OPSET = 13
    
    # Reproducibility
    RANDOM_SEED = 42


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int = Config.RANDOM_SEED):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Random seed set to {seed}")


def setup_directories():
    """Create output directory structure"""
    dirs = [Config.OUTPUT_DIR, Config.MODEL_DIR, 
            Config.RESULTS_DIR, Config.LOGS_DIR]
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directories created at {Config.OUTPUT_DIR}")


def mount_drive_and_unzip(zip_path: str = "/content/drive/MyDrive/Dataset.zip"):
    """Mount Google Drive and extract dataset"""
    from google.colab import drive
    
    # Mount Drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
        print("✓ Google Drive mounted")
    
    # Extract dataset
    if not os.path.exists(Config.DATASET_ROOT):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('/content/')
        print(f"✓ Dataset extracted to {Config.DATASET_ROOT}")
    else:
        print(f"✓ Dataset already exists at {Config.DATASET_ROOT}")


# ============================================================================
# STAGE-AWARE INFERENCE MODULE
# ============================================================================

class StageDetector:
    """
    Rule-based routing system to determine if input is Wafer or Die level.
    Uses image size, repetition patterns, and edge density.
    """
    
    @staticmethod
    def detect_repetition_score(image: np.ndarray) -> float:
        """
        Calculate pattern repetition using autocorrelation.
        High score indicates die-level (repeated structures)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))  # Downsample for speed
        
        # FFT-based autocorrelation
        f = np.fft.fft2(gray)
        power_spectrum = np.abs(f) ** 2
        autocorr = np.fft.ifft2(power_spectrum).real
        
        # Normalize and exclude DC component
        autocorr = autocorr / autocorr.max()
        center = autocorr.shape[0] // 2
        autocorr[center-5:center+5, center-5:center+5] = 0
        
        return float(autocorr.max())
    
    @staticmethod
    def detect_edge_density(image: np.ndarray) -> float:
        """Calculate ratio of edge pixels (die images have more edges)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = np.count_nonzero(edges) / edges.size
        return float(density)
    
    @classmethod
    def classify_stage(cls, image: np.ndarray) -> str:
        """
        Determine if image is WAFER or DIE level
        
        Decision Logic:
        - Large size + low repetition + low edges = WAFER
        - Small size + high repetition + high edges = DIE
        """
        h, w = image.shape[:2]
        min_dim = min(h, w)
        
        # Feature extraction
        repetition = cls.detect_repetition_score(image)
        edge_density = cls.detect_edge_density(image)
        
        # Decision tree
        if min_dim >= Config.WAFER_SIZE_THRESHOLD:
            if repetition < Config.REPETITION_THRESHOLD and \
               edge_density < Config.EDGE_DENSITY_THRESHOLD:
                return "WAFER"
        
        return "DIE"


# ============================================================================
# TILE-BASED PROCESSING
# ============================================================================

class TileProcessor:
    """Sliding window tiling for high-resolution images"""
    
    @staticmethod
    def extract_tiles(
        image: np.ndarray, 
        tile_size: int = Config.TILE_SIZE,
        overlap: int = Config.TILE_OVERLAP
    ) -> List[np.ndarray]:
        """
        Extract overlapping tiles from large image using sliding window
        
        Args:
            image: Input image (H, W, C)
            tile_size: Size of each tile (square)
            overlap: Overlap between adjacent tiles
            
        Returns:
            List of tile images
        """
        h, w = image.shape[:2]
        stride = tile_size - overlap
        tiles = []
        
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
        
        # Handle edge cases (right and bottom borders)
        if h % stride != 0:
            for x in range(0, w - tile_size + 1, stride):
                tile = image[h-tile_size:h, x:x+tile_size]
                tiles.append(tile)
        
        if w % stride != 0:
            for y in range(0, h - tile_size + 1, stride):
                tile = image[y:y+tile_size, w-tile_size:w]
                tiles.append(tile)
        
        return tiles
    
    @staticmethod
    def aggregate_predictions(
        tile_predictions: List[np.ndarray],
        method: str = "max"
    ) -> np.ndarray:
        """
        Aggregate predictions from multiple tiles
        
        Args:
            tile_predictions: List of prediction arrays
            method: 'max' (most confident), 'mean', or 'voting'
            
        Returns:
            Aggregated prediction
        """
        if method == "max":
            # Return prediction with highest confidence
            max_confidences = [pred.max() for pred in tile_predictions]
            return tile_predictions[np.argmax(max_confidences)]
        
        elif method == "mean":
            # Average predictions
            return np.mean(tile_predictions, axis=0)
        
        elif method == "voting":
            # Majority voting
            classes = [np.argmax(pred) for pred in tile_predictions]
            unique, counts = np.unique(classes, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            
            # Create one-hot encoded result
            result = np.zeros_like(tile_predictions[0])
            result[majority_class] = 1.0
            return result
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


# ============================================================================
# DATA PREPROCESSING & AUGMENTATION
# ============================================================================

class IndustrialAugmentation:
    """
    Industrial-safe augmentation pipeline.
    Avoids aggressive transforms that could create unrealistic defects.
    """
    
    @staticmethod
    def get_train_generator(directory: str, batch_size: int) -> ImageDataGenerator:
        """Training data generator with conservative augmentations"""
        datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=5,  # ±5° only
            brightness_range=[0.9, 1.1],
            contrast_range=[0.9, 1.1],
            fill_mode='reflect',
            preprocessing_function=lambda x: x  # Placeholder for custom preprocessing
        )
        
        return datagen.flow_from_directory(
            directory,
            target_size=Config.INPUT_SHAPE[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=Config.RANDOM_SEED
        )
    
    @staticmethod
    def get_val_generator(directory: str, batch_size: int) -> ImageDataGenerator:
        """Validation/Test generator (no augmentation)"""
        datagen = ImageDataGenerator(rescale=1./255)
        
        return datagen.flow_from_directory(
            directory,
            target_size=Config.INPUT_SHAPE[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )


# ============================================================================
# DUAL-HEAD ARCHITECTURE
# ============================================================================

class DualHeadDefectModel:
    """
    Lightweight dual-head architecture with shared MobileNetV2 backbone
    and separate classification heads for Wafer and Die stages.
    """
    
    def __init__(self, wafer_classes: int = 8, die_classes: int = 3):
        self.wafer_classes = wafer_classes
        self.die_classes = die_classes
        self.backbone = None
        self.wafer_model = None
        self.die_model = None
    
    def build_backbone(self) -> models.Model:
        """Create shared MobileNetV2 feature extractor"""
        base_model = MobileNetV2(
            input_shape=Config.INPUT_SHAPE,
            include_top=False,
            weights='imagenet',
            alpha=Config.ALPHA  # Width multiplier for edge optimization
        )
        
        # Freeze backbone initially
        base_model.trainable = False
        
        # Add global pooling
        inputs = keras.Input(shape=Config.INPUT_SHAPE)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        
        self.backbone = models.Model(inputs, x, name="shared_backbone")
        print(f"✓ Backbone created: {Config.BACKBONE} (alpha={Config.ALPHA})")
        return self.backbone
    
    def build_wafer_head(self) -> models.Model:
        """Build Wafer classification head (8 classes)"""
        if self.backbone is None:
            self.build_backbone()
        
        inputs = self.backbone.input
        features = self.backbone.output
        
        # Wafer-specific head
        x = layers.Dense(256, activation='relu', name='wafer_dense1')(features)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu', name='wafer_dense2')(x)
        outputs = layers.Dense(
            self.wafer_classes, 
            activation='softmax', 
            name='wafer_output'
        )(x)
        
        self.wafer_model = models.Model(inputs, outputs, name="wafer_classifier")
        print(f"✓ Wafer head built: {self.wafer_classes} classes")
        return self.wafer_model
    
    def build_die_head(self) -> models.Model:
        """Build Die classification head (3 classes)"""
        if self.backbone is None:
            self.build_backbone()
        
        inputs = self.backbone.input
        features = self.backbone.output
        
        # Die-specific head
        x = layers.Dense(128, activation='relu', name='die_dense1')(features)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu', name='die_dense2')(x)
        outputs = layers.Dense(
            self.die_classes, 
            activation='softmax', 
            name='die_output'
        )(x)
        
        self.die_model = models.Model(inputs, outputs, name="die_classifier")
        print(f"✓ Die head built: {self.die_classes} classes")
        return self.die_model
    
    def enable_fine_tuning(self, model: models.Model, layers_to_unfreeze: int = 30):
        """
        Enable progressive fine-tuning by unfreezing top layers of backbone
        
        Args:
            model: Wafer or Die model
            layers_to_unfreeze: Number of top layers to unfreeze
        """
        # Find backbone layers in the model
        for layer in model.layers:
            if 'mobilenetv2' in layer.name.lower():
                # Unfreeze top N layers
                layer.trainable = True
                for sublayer in layer.layers[:-layers_to_unfreeze]:
                    sublayer.trainable = False
                
                print(f"✓ Fine-tuning enabled: {layers_to_unfreeze} layers unfrozen")
                return
    
    def compile_model(
        self, 
        model: models.Model, 
        learning_rate: float = Config.INITIAL_LR
    ):
        """Compile model with optimizer and loss"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        print(f"✓ Model compiled with LR={learning_rate}")


# ============================================================================
# ADAPTIVE SAMPLING FOR CLASS IMBALANCE
# ============================================================================

class AdaptiveSampler:
    """Compute class weights to handle imbalanced datasets"""
    
    @staticmethod
    def compute_weights(train_generator) -> Dict[int, float]:
        """
        Calculate class weights based on inverse frequency
        
        Args:
            train_generator: Keras ImageDataGenerator
            
        Returns:
            Dictionary mapping class indices to weights
        """
        # Get all labels
        labels = train_generator.classes
        
        # Compute weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        weight_dict = dict(enumerate(class_weights))
        
        print("✓ Class weights computed:")
        for idx, weight in weight_dict.items():
            class_name = list(train_generator.class_indices.keys())[idx]
            print(f"  {class_name}: {weight:.3f}")
        
        return weight_dict


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class TrainingPipeline:
    """Complete training workflow with callbacks and fine-tuning"""
    
    def __init__(self, model: models.Model, stage: str):
        self.model = model
        self.stage = stage  # 'wafer' or 'die'
        self.history = None
    
    def get_callbacks(self) -> List:
        """Create training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=f"{Config.MODEL_DIR}/{self.stage}_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce LR on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=f"{Config.LOGS_DIR}/{self.stage}_{timestamp}",
                histogram_freq=1
            ),
            
            # CSV logger
            CSVLogger(
                filename=f"{Config.RESULTS_DIR}/{self.stage}_training.csv"
            )
        ]
        
        return callbacks
    
    def train(
        self,
        train_gen,
        val_gen,
        class_weights: Optional[Dict] = None,
        fine_tune_epoch: int = Config.FINE_TUNE_EPOCH
    ):
        """
        Execute two-stage training: initial + fine-tuning
        
        Args:
            train_gen: Training data generator
            val_gen: Validation data generator
            class_weights: Class weight dictionary for imbalanced data
            fine_tune_epoch: Epoch to start fine-tuning
        """
        print(f"\n{'='*80}")
        print(f"Training {self.stage.upper()} Model")
        print(f"{'='*80}\n")
        
        # Stage 1: Train with frozen backbone
        print("Stage 1: Training classification head...")
        history1 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=fine_tune_epoch,
            callbacks=self.get_callbacks(),
            class_weight=class_weights,
            verbose=1
        )
        
        # Stage 2: Fine-tune with unfrozen backbone
        print(f"\nStage 2: Fine-tuning (unfreezing backbone)...")
        dual_model = DualHeadDefectModel()
        dual_model.enable_fine_tuning(self.model)
        dual_model.compile_model(self.model, learning_rate=Config.FINE_TUNE_LR)
        
        history2 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            initial_epoch=fine_tune_epoch,
            epochs=Config.EPOCHS,
            callbacks=self.get_callbacks(),
            class_weight=class_weights,
            verbose=1
        )
        
        # Merge histories
        self.history = self._merge_histories(history1, history2)
        
        print(f"\n✓ {self.stage.upper()} training complete!")
        return self.history
    
    @staticmethod
    def _merge_histories(h1, h2):
        """Combine two History objects"""
        merged = {}
        for key in h1.history.keys():
            merged[key] = h1.history[key] + h2.history[key]
        return merged
    
    def plot_training_curves(self, save_path: str = None):
        """Visualize training metrics"""
        if self.history is None:
            print("⚠ No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.stage.upper()} Model Training Curves', fontsize=16)
        
        metrics = [
            ('accuracy', 'Accuracy'),
            ('loss', 'Loss'),
            ('precision', 'Precision'),
            ('recall', 'Recall')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            if metric in self.history:
                ax.plot(self.history[metric], label=f'Train {title}')
            if f'val_{metric}' in self.history:
                ax.plot(self.history[f'val_{metric}'], label=f'Val {title}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(f'{title} over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training curves saved to {save_path}")
        
        plt.show()


# ============================================================================
# EVALUATION & METRICS
# ============================================================================

class ModelEvaluator:
    """Comprehensive evaluation with industrial metrics"""
    
    def __init__(self, model: models.Model, stage: str, class_names: List[str]):
        self.model = model
        self.stage = stage
        self.class_names = class_names
    
    def evaluate(self, test_generator, confidence_threshold: float = Config.CONFIDENCE_THRESHOLD):
        """
        Complete evaluation pipeline with metrics and visualizations
        
        Args:
            test_generator: Test data generator
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dictionary containing all metrics
        """
        print(f"\n{'='*80}")
        print(f"Evaluating {self.stage.upper()} Model")
        print(f"{'='*80}\n")
        
        # Get predictions
        print("Running inference on test set...")
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        # Apply confidence threshold (route low confidence to "Other/Unknown")
        low_confidence_mask = confidences < confidence_threshold
        if self.stage == 'wafer':
            other_class_idx = self.class_names.index('Other')
            predicted_classes[low_confidence_mask] = other_class_idx
        elif self.stage == 'die':
            unknown_class_idx = self.class_names.index('Unknown')
            predicted_classes[low_confidence_mask] = unknown_class_idx
        
        true_classes = test_generator.classes
        
        # Calculate metrics
        metrics = self._calculate_metrics(true_classes, predicted_classes, confidences)
        
        # Generate visualizations
        self._plot_confusion_matrix(true_classes, predicted_classes)
        self._plot_confidence_distribution(confidences)
        
        # Benchmark inference latency
        latency = self._benchmark_latency(test_generator)
        metrics['avg_latency_ms'] = latency
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, confidences):
        """Calculate precision, recall, F1-score"""
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names))
        )
        
        # Overall metrics
        accuracy = np.mean(y_true == y_pred)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4
        ))
        
        metrics = {
            'accuracy': accuracy,
            'per_class_precision': dict(zip(self.class_names, precision)),
            'per_class_recall': dict(zip(self.class_names, recall)),
            'per_class_f1': dict(zip(self.class_names, f1)),
            'per_class_support': dict(zip(self.class_names, support)),
            'avg_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences))
        }
        
        return metrics
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Generate confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'{self.stage.upper()} Model - Confusion Matrix', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = f"{Config.RESULTS_DIR}/{self.stage}_confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.show()
    
    def _plot_confidence_distribution(self, confidences):
        """Plot distribution of prediction confidences"""
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(
            Config.CONFIDENCE_THRESHOLD, 
            color='red', linestyle='--', 
            label=f'Threshold = {Config.CONFIDENCE_THRESHOLD}'
        )
        plt.axvline(
            Config.EARLY_EXIT_CONFIDENCE,
            color='green', linestyle='--',
            label=f'Early Exit = {Config.EARLY_EXIT_CONFIDENCE}'
        )
        plt.xlabel('Prediction Confidence', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'{self.stage.upper()} Model - Confidence Distribution', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = f"{Config.RESULTS_DIR}/{self.stage}_confidence_dist.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confidence distribution saved to {save_path}")
        plt.show()
    
    def _benchmark_latency(self, test_generator, num_samples: int = 100):
        """Measure inference latency for edge deployment validation"""
        print("\nBenchmarking inference latency...")
        
        # Get random samples
        sample_images = []
        for _ in range(num_samples):
            batch = next(test_generator)
            sample_images.append(batch[0][0:1])  # Take first image from batch
        
        # Warmup
        for _ in range(10):
            self.model.predict(sample_images[0], verbose=0)
        
        # Measure
        latencies = []
        for img in sample_images:
            start = time.time()
            _ = self.model.predict(img, verbose=0)
            latencies.append((time.time() - start) * 1000)  # Convert to ms
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        print(f"  Average Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
        print(f"  Min Latency: {np.min(latencies):.2f} ms")
        print(f"  Max Latency: {np.max(latencies):.2f} ms")
        
        return avg_latency
    
    def _save_results(self, metrics: Dict):
        """Save evaluation results to JSON"""
        results = {
            'stage': self.stage,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        save_path = f"{Config.RESULTS_DIR}/{self.stage}_metrics.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✓ Metrics saved to {save_path}")


# ============================================================================
# ONNX EXPORT FOR EDGE DEPLOYMENT
# ============================================================================

class ONNXExporter:
    """Export trained models to ONNX format for NXP eIQ deployment"""
    
    @staticmethod
    def export_model(
        model: models.Model,
        stage: str,
        opset: int = Config.ONNX_OPSET,
        quantize: bool = Config.QUANTIZATION
    ):
        """
        Convert Keras model to ONNX format
        
        Args:
            model: Trained Keras model
            stage: 'wafer' or 'die'
            opset: ONNX opset version
            quantize: Apply dynamic quantization for edge optimization
        """
        print(f"\n{'='*80}")
        print(f"Exporting {stage.upper()} Model to ONNX")
        print(f"{'='*80}\n")
        
        # Define output path
        onnx_path = f"{Config.MODEL_DIR}/{stage}_model.onnx"
        
        # Convert to ONNX
        spec = (tf.TensorSpec(model.input.shape, tf.float32, name="input"),)
        
        try:
            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=spec,
                opset=opset,
                output_path=onnx_path
            )
            
            print(f"✓ ONNX model saved to {onnx_path}")
            
            # Verify model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model verification passed")
            
            # Report model size
            model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"  Model Size: {model_size_mb:.2f} MB")
            
            # Quantization (optional)
            if quantize:
                print("\n⚠ Note: Dynamic quantization requires ONNX Runtime")
                print("  For production deployment, use NXP eIQ Toolkit for INT8 quantization")
            
            return onnx_path
            
        except Exception as e:
            print(f"✗ ONNX export failed: {str(e)}")
            return None


# ============================================================================
# END-TO-END INFERENCE PIPELINE
# ============================================================================

class InferencePipeline:
    """Production inference with stage detection and confidence handling"""
    
    def __init__(self, wafer_model_path: str, die_model_path: str):
        print("Loading models for inference...")
        self.wafer_model = keras.models.load_model(wafer_model_path)
        self.die_model = keras.models.load_model(die_model_path)
        self.stage_detector = StageDetector()
        self.tile_processor = TileProcessor()
        print("✓ Models loaded successfully")
    
    def predict(
        self, 
        image_path: str,
        use_tiling: bool = True,
        early_exit: bool = True
    ) -> Dict:
        """
        Complete inference pipeline with all novelty features
        
        Args:
            image_path: Path to input image
            use_tiling: Apply tile-based processing for large images
            early_exit: Enable early exit for high-confidence predictions
            
        Returns:
            Prediction dictionary with class, confidence, and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Stage detection
        stage = self.stage_detector.classify_stage(image_rgb)
        model = self.wafer_model if stage == "WAFER" else self.die_model
        class_names = Config.WAFER_CLASSES if stage == "WAFER" else Config.DIE_CLASSES
        
        # Tile-based processing (if enabled and image is large)
        h, w = image_rgb.shape[:2]
        if use_tiling and min(h, w) > Config.TILE_SIZE * 2:
            tiles = self.tile_processor.extract_tiles(image_rgb)
            
            # Preprocess tiles
            tiles_processed = np.array([
                cv2.resize(tile, Config.INPUT_SHAPE[:2]) / 255.0 
                for tile in tiles
            ])
            
            # Predict on tiles
            tile_preds = model.predict(tiles_processed, verbose=0)
            
            # Aggregate
            prediction = self.tile_processor.aggregate_predictions(
                list(tile_preds), method="max"
            )
        else:
            # Direct prediction
            img_resized = cv2.resize(image_rgb, Config.INPUT_SHAPE[:2])
            img_normalized = np.expand_dims(img_resized / 255.0, axis=0)
            prediction = model.predict(img_normalized, verbose=0)[0]
        
        # Extract results
        predicted_class_idx = int(np.argmax(prediction))
        confidence = float(prediction[predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        # Confidence-based handling
        if confidence < Config.CONFIDENCE_THRESHOLD:
            if stage == "WAFER":
                predicted_class = "Other"
            else:
                predicted_class = "Unknown"
        
        # Early exit simulation
        early_exit_triggered = early_exit and confidence >= Config.EARLY_EXIT_CONFIDENCE
        
        result = {
            'stage': stage,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': dict(zip(class_names, prediction.tolist())),
            'early_exit': early_exit_triggered,
            'tiling_used': use_tiling and min(h, w) > Config.TILE_SIZE * 2,
            'image_size': (h, w)
        }
        
        return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Complete training and evaluation pipeline"""
    
    # Setup
    set_seed()
    setup_directories()
    
    # Mount Drive and extract data (uncomment for Colab)
    # mount_drive_and_unzip()
    
    # Initialize architecture
    dual_model = DualHeadDefectModel(
        wafer_classes=len(Config.WAFER_CLASSES),
        die_classes=len(Config.DIE_CLASSES)
    )
    
    # ========================================================================
    # WAFER MODEL TRAINING
    # ========================================================================
    
    print("\n" + "="*80)
    print("WAFER MODEL PIPELINE")
    print("="*80 + "\n")
    
    # Build model
    wafer_model = dual_model.build_wafer_head()
    dual_model.compile_model(wafer_model)
    
    # Data generators
    augmenter = IndustrialAugmentation()
    train_gen_wafer = augmenter.get_train_generator(
        Config.TRAIN_DIR, Config.BATCH_SIZE
    )
    val_gen_wafer = augmenter.get_val_generator(
        Config.VAL_DIR, Config.BATCH_SIZE
    )
    test_gen_wafer = augmenter.get_val_generator(
        Config.TEST_DIR, Config.BATCH_SIZE
    )
    
    # Adaptive sampling
    sampler = AdaptiveSampler()
    class_weights_wafer = sampler.compute_weights(train_gen_wafer)
    
    # Train
    trainer_wafer = TrainingPipeline(wafer_model, stage='wafer')
    trainer_wafer.train(
        train_gen_wafer, 
        val_gen_wafer,
        class_weights=class_weights_wafer
    )
    
    # Visualize training
    trainer_wafer.plot_training_curves(
        save_path=f"{Config.RESULTS_DIR}/wafer_training_curves.png"
    )
    
    # Evaluate
    evaluator_wafer = ModelEvaluator(
        wafer_model, 
        stage='wafer',
        class_names=Config.WAFER_CLASSES
    )
    wafer_metrics = evaluator_wafer.evaluate(test_gen_wafer)
    
    # Export to ONNX
    exporter = ONNXExporter()
    exporter.export_model(wafer_model, stage='wafer')
    
    # ========================================================================
    # DIE MODEL TRAINING
    # ========================================================================
    
    print("\n" + "="*80)
    print("DIE MODEL PIPELINE")
    print("="*80 + "\n")
    
    # Build model (will reuse backbone)
    die_model = dual_model.build_die_head()
    dual_model.compile_model(die_model)
    
    # Data generators (assuming die images are in separate directories)
    # If die images are mixed with wafer, you'll need to reorganize
    train_gen_die = augmenter.get_train_generator(
        Config.TRAIN_DIR, Config.BATCH_SIZE
    )
    val_gen_die = augmenter.get_val_generator(
        Config.VAL_DIR, Config.BATCH_SIZE
    )
    test_gen_die = augmenter.get_val_generator(
        Config.TEST_DIR, Config.BATCH_SIZE
    )
    
    # Adaptive sampling
    class_weights_die = sampler.compute_weights(train_gen_die)
    
    # Train
    trainer_die = TrainingPipeline(die_model, stage='die')
    trainer_die.train(
        train_gen_die,
        val_gen_die,
        class_weights=class_weights_die
    )
    
    # Visualize training
    trainer_die.plot_training_curves(
        save_path=f"{Config.RESULTS_DIR}/die_training_curves.png"
    )
    
    # Evaluate
    evaluator_die = ModelEvaluator(
        die_model,
        stage='die',
        class_names=Config.DIE_CLASSES
    )
    die_metrics = evaluator_die.evaluate(test_gen_die)
    
    # Export to ONNX
    exporter.export_model(die_model, stage='die')
    
    # ========================================================================
    # UNIFIED INFERENCE DEMO
    # ========================================================================
    
    print("\n" + "="*80)
    print("UNIFIED INFERENCE PIPELINE DEMO")
    print("="*80 + "\n")
    
    # Create inference pipeline
    pipeline = InferencePipeline(
        wafer_model_path=f"{Config.MODEL_DIR}/wafer_best.h5",
        die_model_path=f"{Config.MODEL_DIR}/die_best.h5"
    )
    
    # Demo prediction (replace with actual test image path)
    # test_image = f"{Config.TEST_DIR}/Center/sample_image.jpg"
    # result = pipeline.predict(test_image)
    # print(json.dumps(result, indent=2))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE! 🎉")
    print("="*80)
    print(f"\nOutputs saved to: {Config.OUTPUT_DIR}")
    print(f"  - Models: {Config.MODEL_DIR}")
    print(f"  - Results: {Config.RESULTS_DIR}")
    print(f"  - Logs: {Config.LOGS_DIR}")
    print("\nNext Steps:")
    print("  1. Review metrics in JSON files")
    print("  2. Validate ONNX models with NXP eIQ Toolkit")
    print("  3. Deploy to edge device and benchmark real-world performance")
    print("="*80)


if __name__ == "__main__":
    main()
