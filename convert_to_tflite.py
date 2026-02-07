"""
TFLite Converter for NXP eIQ Edge Deployment
=============================================

Converts trained Keras models to optimized TFLite format with INT8 quantization
for maximum performance on NXP i.MX RT and i.MX 8M platforms.

Usage:
    python convert_to_tflite.py --model wafer_best.h5 --output wafer_int8.tflite
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional


class TFLiteConverter:
    """
    Advanced TFLite conversion with quantization support
    """
    
    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        self.model = None
        
    def load_model(self):
        """Load Keras model"""
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("✓ Model loaded successfully")
        
    def representative_dataset_gen(self, num_samples: int = 100):
        """
        Generate representative dataset for full integer quantization.
        Uses random data - in production, use real calibration images!
        
        Args:
            num_samples: Number of calibration samples
            
        Yields:
            Sample inputs for quantization calibration
        """
        input_shape = self.model.input.shape[1:]
        
        for _ in range(num_samples):
            # Generate random input (replace with real data for better accuracy)
            data = np.random.random((1, *input_shape)).astype(np.float32)
            yield [data]
    
    def convert_float32(self):
        """
        Convert to FP32 TFLite (no quantization)
        
        Returns:
            Path to converted model
        """
        print("\nConverting to FP32 TFLite...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimizations
        converter.optimizations = []  # No quantization
        
        tflite_model = converter.convert()
        
        # Save
        output_fp32 = self.output_path.replace('.tflite', '_fp32.tflite')
        with open(output_fp32, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✓ FP32 model saved to {output_fp32} ({size_mb:.2f} MB)")
        
        return output_fp32
    
    def convert_float16(self):
        """
        Convert to FP16 TFLite (half precision)
        
        Returns:
            Path to converted model
        """
        print("\nConverting to FP16 TFLite...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Enable FP16 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save
        output_fp16 = self.output_path.replace('.tflite', '_fp16.tflite')
        with open(output_fp16, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✓ FP16 model saved to {output_fp16} ({size_mb:.2f} MB)")
        
        return output_fp16
    
    def convert_int8(self, use_representative_data: bool = True):
        """
        Convert to INT8 TFLite (full integer quantization)
        Recommended for NXP eIQ NPU acceleration
        
        Args:
            use_representative_data: Use calibration data for quantization
            
        Returns:
            Path to converted model
        """
        print("\nConverting to INT8 TFLite (quantized)...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Enable full integer quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if use_representative_data:
            print("  Using representative dataset for calibration...")
            converter.representative_dataset = self.representative_dataset_gen
        
        # Force INT8 for inputs/outputs
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        try:
            tflite_model = converter.convert()
            
            # Save
            with open(self.output_path, 'wb') as f:
                f.write(tflite_model)
            
            size_mb = len(tflite_model) / (1024 * 1024)
            print(f"✓ INT8 model saved to {self.output_path} ({size_mb:.2f} MB)")
            
            return self.output_path
            
        except Exception as e:
            print(f"✗ INT8 conversion failed: {e}")
            print("  Falling back to dynamic range quantization...")
            
            # Fallback: dynamic range quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            with open(self.output_path, 'wb') as f:
                f.write(tflite_model)
            
            size_mb = len(tflite_model) / (1024 * 1024)
            print(f"✓ Dynamic quantized model saved ({size_mb:.2f} MB)")
            
            return self.output_path
    
    def validate_model(self, tflite_path: str):
        """
        Validate converted TFLite model
        
        Args:
            tflite_path: Path to TFLite model
        """
        print(f"\nValidating {tflite_path}...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("  Input Details:")
        print(f"    Shape: {input_details[0]['shape']}")
        print(f"    Type: {input_details[0]['dtype']}")
        
        print("  Output Details:")
        print(f"    Shape: {output_details[0]['shape']}")
        print(f"    Type: {output_details[0]['dtype']}")
        
        # Test inference
        input_shape = input_details[0]['shape']
        test_input = np.random.random(input_shape).astype(input_details[0]['dtype'])
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"  Test inference successful! Output shape: {output.shape}")
        print("✓ Validation passed")
    
    def convert_all(self):
        """
        Convert model to all TFLite formats
        
        Returns:
            Dictionary of converted model paths
        """
        self.load_model()
        
        results = {
            'fp32': self.convert_float32(),
            'fp16': self.convert_float16(),
            'int8': self.convert_int8()
        }
        
        # Validate INT8 model
        self.validate_model(results['int8'])
        
        # Compare sizes
        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        
        for format_name, path in results.items():
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"{format_name.upper()}: {size_mb:.2f} MB - {path}")
        
        print("="*60)
        print("\nRecommendation for NXP eIQ:")
        print(f"  Use INT8 model: {results['int8']}")
        print("  Expected speedup: 2-4x vs FP32")
        print("  Memory reduction: ~75%")
        
        return results


def main():
    """CLI interface for TFLite conversion"""
    
    parser = argparse.ArgumentParser(
        description="Convert Keras models to TFLite for NXP eIQ deployment"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to Keras model (.h5 or SavedModel)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for TFLite model (.tflite)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['int8', 'fp16', 'fp32', 'all'],
        default='int8',
        help='Quantization format (default: int8)'
    )
    
    parser.add_argument(
        '--calibration-samples',
        type=int,
        default=100,
        help='Number of samples for INT8 calibration (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = TFLiteConverter(args.model, args.output)
    
    # Convert based on format
    if args.format == 'all':
        converter.convert_all()
    else:
        converter.load_model()
        
        if args.format == 'int8':
            path = converter.convert_int8()
        elif args.format == 'fp16':
            path = converter.convert_float16()
        elif args.format == 'fp32':
            path = converter.convert_float32()
        
        converter.validate_model(path)
    
    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
