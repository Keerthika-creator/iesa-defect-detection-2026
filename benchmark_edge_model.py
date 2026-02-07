"""
Edge Deployment Benchmark Suite
================================

Comprehensive performance benchmarking for TFLite models on edge devices.
Measures latency, throughput, memory usage, and accuracy metrics.

Usage:
    python benchmark_edge_model.py --model wafer_int8.tflite --test-dir Dataset/test
"""

import argparse
import time
import psutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2


class EdgeBenchmark:
    """
    Benchmark TFLite models for edge deployment validation
    """
    
    def __init__(self, model_path: str, test_dir: str = None):
        self.model_path = model_path
        self.test_dir = test_dir
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
    def load_model(self):
        """Load TFLite model and allocate tensors"""
        print(f"Loading TFLite model: {self.model_path}")
        
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print("✓ Model loaded")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Input type: {self.input_details[0]['dtype']}")
        print(f"  Output shape: {self.output_details[0]['shape']}")
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Get expected input shape
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((width, height))
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize
        img_array = img_array / 255.0
        
        # Quantize if model expects INT8
        if self.input_details[0]['dtype'] == np.int8:
            input_scale = self.input_details[0]['quantization'][0]
            input_zero_point = self.input_details[0]['quantization'][1]
            img_array = img_array / input_scale + input_zero_point
            img_array = np.clip(img_array, -128, 127).astype(np.int8)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """
        Execute model inference
        
        Args:
            input_data: Preprocessed input array
            
        Returns:
            Model output
        """
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize if output is INT8
        if self.output_details[0]['dtype'] == np.int8:
            output_scale = self.output_details[0]['quantization'][0]
            output_zero_point = self.output_details[0]['quantization'][1]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        return output
    
    def benchmark_latency(self, num_iterations: int = 1000) -> Dict:
        """
        Measure inference latency
        
        Args:
            num_iterations: Number of inference runs
            
        Returns:
            Latency statistics
        """
        print(f"\nBenchmarking latency ({num_iterations} iterations)...")
        
        # Generate random input
        input_shape = self.input_details[0]['shape']
        input_dtype = self.input_details[0]['dtype']
        
        # Warmup
        print("  Warming up...")
        for _ in range(10):
            test_input = np.random.random(input_shape).astype(np.float32)
            if input_dtype == np.int8:
                test_input = (test_input * 255 - 128).astype(np.int8)
            self.run_inference(test_input)
        
        # Measure
        print("  Measuring...")
        latencies = []
        
        for i in range(num_iterations):
            test_input = np.random.random(input_shape).astype(np.float32)
            if input_dtype == np.int8:
                test_input = (test_input * 255 - 128).astype(np.int8)
            
            start = time.perf_counter()
            self.run_inference(test_input)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
            
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{num_iterations}")
        
        # Calculate statistics
        stats = {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'median_ms': float(np.median(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'throughput_fps': 1000.0 / np.mean(latencies)
        }
        
        print("\n  Latency Results:")
        print(f"    Mean:   {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
        print(f"    Median: {stats['median_ms']:.2f} ms")
        print(f"    Min:    {stats['min_ms']:.2f} ms")
        print(f"    Max:    {stats['max_ms']:.2f} ms")
        print(f"    P95:    {stats['p95_ms']:.2f} ms")
        print(f"    P99:    {stats['p99_ms']:.2f} ms")
        print(f"    Throughput: {stats['throughput_fps']:.1f} FPS")
        
        return stats, latencies
    
    def benchmark_memory(self) -> Dict:
        """
        Measure memory footprint
        
        Returns:
            Memory statistics
        """
        print("\nBenchmarking memory usage...")
        
        # Model file size
        model_size_mb = Path(self.model_path).stat().st_size / (1024 * 1024)
        
        # Runtime memory (approximate via tensor sizes)
        input_size = np.prod(self.input_details[0]['shape']) * \
                     self.input_details[0]['dtype'].itemsize
        output_size = np.prod(self.output_details[0]['shape']) * \
                      self.output_details[0]['dtype'].itemsize
        
        # Process memory
        process = psutil.Process()
        mem_info = process.memory_info()
        
        stats = {
            'model_size_mb': float(model_size_mb),
            'input_tensor_bytes': int(input_size),
            'output_tensor_bytes': int(output_size),
            'process_rss_mb': mem_info.rss / (1024 * 1024),
            'process_vms_mb': mem_info.vms / (1024 * 1024)
        }
        
        print("  Memory Results:")
        print(f"    Model Size:  {stats['model_size_mb']:.2f} MB")
        print(f"    Input Size:  {stats['input_tensor_bytes'] / 1024:.2f} KB")
        print(f"    Output Size: {stats['output_tensor_bytes'] / 1024:.2f} KB")
        print(f"    Process RSS: {stats['process_rss_mb']:.2f} MB")
        
        return stats
    
    def benchmark_accuracy(self, class_names: List[str]) -> Dict:
        """
        Measure prediction accuracy on test set
        
        Args:
            class_names: List of class names
            
        Returns:
            Accuracy statistics
        """
        if self.test_dir is None:
            print("⚠ Test directory not provided, skipping accuracy benchmark")
            return {}
        
        print(f"\nBenchmarking accuracy on {self.test_dir}...")
        
        # Collect test images
        test_images = []
        true_labels = []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = Path(self.test_dir) / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*.jpg'):
                test_images.append(str(img_path))
                true_labels.append(class_idx)
        
        print(f"  Found {len(test_images)} test images")
        
        # Run inference
        predictions = []
        confidences = []
        
        for i, img_path in enumerate(test_images):
            input_data = self.preprocess_image(img_path)
            output = self.run_inference(input_data)[0]
            
            # Softmax if needed
            if output.max() > 1.0:
                exp_out = np.exp(output - output.max())
                output = exp_out / exp_out.sum()
            
            pred_class = int(np.argmax(output))
            confidence = float(output[pred_class])
            
            predictions.append(pred_class)
            confidences.append(confidence)
            
            if (i + 1) % 50 == 0:
                print(f"    Progress: {i+1}/{len(test_images)}")
        
        # Calculate metrics
        correct = sum([p == t for p, t in zip(predictions, true_labels)])
        accuracy = correct / len(true_labels)
        
        stats = {
            'num_samples': len(test_images),
            'accuracy': float(accuracy),
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences))
        }
        
        print("\n  Accuracy Results:")
        print(f"    Accuracy:    {accuracy:.2%}")
        print(f"    Confidence:  {stats['mean_confidence']:.2%} ± {stats['std_confidence']:.2%}")
        
        return stats, predictions, true_labels, confidences
    
    def plot_latency_distribution(self, latencies: List[float], save_path: str):
        """
        Visualize latency distribution
        
        Args:
            latencies: List of latency measurements
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(latencies, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(latencies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(latencies):.2f} ms')
        plt.axvline(np.median(latencies), color='green', linestyle='--',
                   label=f'Median: {np.median(latencies):.2f} ms')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(latencies, vert=True)
        plt.ylabel('Latency (ms)')
        plt.title('Latency Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Latency plot saved to {save_path}")
        plt.close()
    
    def generate_report(
        self,
        latency_stats: Dict,
        memory_stats: Dict,
        accuracy_stats: Dict,
        save_path: str
    ):
        """
        Generate comprehensive benchmark report
        
        Args:
            latency_stats: Latency metrics
            memory_stats: Memory metrics
            accuracy_stats: Accuracy metrics
            save_path: Path to save JSON report
        """
        report = {
            'model_path': self.model_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'latency': latency_stats,
            'memory': memory_stats,
            'accuracy': accuracy_stats,
            'deployment_readiness': self._assess_deployment_readiness(
                latency_stats, memory_stats, accuracy_stats
            )
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Benchmark report saved to {save_path}")
        
        return report
    
    def _assess_deployment_readiness(
        self,
        latency_stats: Dict,
        memory_stats: Dict,
        accuracy_stats: Dict
    ) -> Dict:
        """
        Assess if model meets edge deployment criteria
        
        Args:
            latency_stats: Latency metrics
            memory_stats: Memory metrics
            accuracy_stats: Accuracy metrics
            
        Returns:
            Deployment assessment
        """
        # Criteria for industrial edge deployment
        LATENCY_TARGET_MS = 50  # 20 FPS minimum
        MEMORY_TARGET_MB = 10   # Small footprint
        ACCURACY_TARGET = 0.85  # 85% minimum accuracy
        
        checks = {
            'latency_ok': latency_stats['mean_ms'] < LATENCY_TARGET_MS,
            'memory_ok': memory_stats['model_size_mb'] < MEMORY_TARGET_MB,
            'accuracy_ok': accuracy_stats.get('accuracy', 0) > ACCURACY_TARGET
        }
        
        return {
            'ready_for_deployment': all(checks.values()),
            'checks': checks,
            'recommendations': self._generate_recommendations(checks, latency_stats, memory_stats)
        }
    
    def _generate_recommendations(
        self,
        checks: Dict,
        latency_stats: Dict,
        memory_stats: Dict
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if not checks['latency_ok']:
            recommendations.append(
                f"⚠ Latency too high ({latency_stats['mean_ms']:.2f} ms). "
                "Consider: pruning, quantization, or hardware acceleration."
            )
        
        if not checks['memory_ok']:
            recommendations.append(
                f"⚠ Model too large ({memory_stats['model_size_mb']:.2f} MB). "
                "Consider: further quantization or knowledge distillation."
            )
        
        if not checks['accuracy_ok']:
            recommendations.append(
                "⚠ Accuracy below target. Consider: more training data, "
                "better augmentation, or ensemble methods."
            )
        
        if all(checks.values()):
            recommendations.append("✅ Model meets all deployment criteria!")
        
        return recommendations


def main():
    """CLI interface for benchmarking"""
    
    parser = argparse.ArgumentParser(
        description="Benchmark TFLite models for edge deployment"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to TFLite model'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default=None,
        help='Path to test dataset directory'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of latency benchmark iterations (default: 1000)'
    )
    
    parser.add_argument(
        '--class-names',
        type=str,
        nargs='+',
        default=['Clean', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Scratch', 'Other'],
        help='List of class names for accuracy evaluation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='Output directory for results (default: benchmark_results)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize benchmark
    benchmark = EdgeBenchmark(args.model, args.test_dir)
    benchmark.load_model()
    
    # Run benchmarks
    print("\n" + "="*60)
    print("EDGE DEPLOYMENT BENCHMARK")
    print("="*60)
    
    # Latency
    latency_stats, latencies = benchmark.benchmark_latency(args.iterations)
    benchmark.plot_latency_distribution(
        latencies,
        str(output_dir / 'latency_distribution.png')
    )
    
    # Memory
    memory_stats = benchmark.benchmark_memory()
    
    # Accuracy
    accuracy_stats = {}
    if args.test_dir:
        accuracy_stats, preds, labels, confs = benchmark.benchmark_accuracy(args.class_names)
    
    # Generate report
    report = benchmark.generate_report(
        latency_stats,
        memory_stats,
        accuracy_stats,
        str(output_dir / 'benchmark_report.json')
    )
    
    # Print deployment assessment
    print("\n" + "="*60)
    print("DEPLOYMENT READINESS ASSESSMENT")
    print("="*60)
    
    readiness = report['deployment_readiness']
    print(f"Ready for Deployment: {readiness['ready_for_deployment']}")
    print("\nChecks:")
    for check, status in readiness['checks'].items():
        symbol = "✅" if status else "❌"
        print(f"  {symbol} {check}: {status}")
    
    print("\nRecommendations:")
    for rec in readiness['recommendations']:
        print(f"  {rec}")
    
    print("="*60)
    print(f"\n✅ Benchmark complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
