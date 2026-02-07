"""
Results Visualization & Analysis Suite
=======================================

Generate publication-quality plots and comprehensive analysis reports
for the defect detection system.

Usage:
    python visualize_results.py --results-dir outputs/results
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class ResultsVisualizer:
    """Comprehensive visualization suite for model results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.wafer_metrics = None
        self.die_metrics = None
        self.wafer_training = None
        self.die_training = None
        
    def load_metrics(self):
        """Load all metric files"""
        print("Loading metrics...")
        
        # JSON metrics
        with open(self.results_dir / 'wafer_metrics.json', 'r') as f:
            self.wafer_metrics = json.load(f)
        
        with open(self.results_dir / 'die_metrics.json', 'r') as f:
            self.die_metrics = json.load(f)
        
        # Training logs
        self.wafer_training = pd.read_csv(self.results_dir / 'wafer_training.csv')
        self.die_training = pd.read_csv(self.results_dir / 'die_training.csv')
        
        print("✓ Metrics loaded successfully")
    
    def plot_training_comparison(self, save_path: str):
        """
        Compare training curves for Wafer and Die models
        
        Args:
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Comparison: Wafer vs Die Models', fontsize=16, fontweight='bold')
        
        metrics = [
            ('loss', 'Loss', axes[0, 0]),
            ('accuracy', 'Accuracy', axes[0, 1]),
            ('precision', 'Precision', axes[1, 0]),
            ('recall', 'Recall', axes[1, 1])
        ]
        
        for metric, title, ax in metrics:
            # Wafer
            if metric in self.wafer_training.columns:
                ax.plot(self.wafer_training[metric], 
                       label='Wafer Train', color='#2E86AB', linewidth=2)
            if f'val_{metric}' in self.wafer_training.columns:
                ax.plot(self.wafer_training[f'val_{metric}'], 
                       label='Wafer Val', color='#2E86AB', linestyle='--', linewidth=2)
            
            # Die
            if metric in self.die_training.columns:
                ax.plot(self.die_training[metric], 
                       label='Die Train', color='#A23B72', linewidth=2)
            if f'val_{metric}' in self.die_training.columns:
                ax.plot(self.die_training[f'val_{metric}'], 
                       label='Die Val', color='#A23B72', linestyle='--', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(f'{title} over Training', fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Training comparison saved to {save_path}")
        plt.close()
    
    def plot_performance_metrics(self, save_path: str):
        """
        Bar chart comparing performance metrics
        
        Args:
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Wafer model
        wafer_metrics_data = self.wafer_metrics['metrics']
        wafer_classes = list(wafer_metrics_data['per_class_precision'].keys())
        wafer_precision = list(wafer_metrics_data['per_class_precision'].values())
        wafer_recall = list(wafer_metrics_data['per_class_recall'].values())
        wafer_f1 = list(wafer_metrics_data['per_class_f1'].values())
        
        x = np.arange(len(wafer_classes))
        width = 0.25
        
        axes[0].bar(x - width, wafer_precision, width, label='Precision', color='#06A77D')
        axes[0].bar(x, wafer_recall, width, label='Recall', color='#D5573B')
        axes[0].bar(x + width, wafer_f1, width, label='F1-Score', color='#F0B67F')
        
        axes[0].set_xlabel('Class', fontsize=11)
        axes[0].set_ylabel('Score', fontsize=11)
        axes[0].set_title('Wafer Model - Per-Class Performance', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(wafer_classes, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1.1])
        
        # Die model
        die_metrics_data = self.die_metrics['metrics']
        die_classes = list(die_metrics_data['per_class_precision'].keys())
        die_precision = list(die_metrics_data['per_class_precision'].values())
        die_recall = list(die_metrics_data['per_class_recall'].values())
        die_f1 = list(die_metrics_data['per_class_f1'].values())
        
        x_die = np.arange(len(die_classes))
        
        axes[1].bar(x_die - width, die_precision, width, label='Precision', color='#06A77D')
        axes[1].bar(x_die, die_recall, width, label='Recall', color='#D5573B')
        axes[1].bar(x_die + width, die_f1, width, label='F1-Score', color='#F0B67F')
        
        axes[1].set_xlabel('Class', fontsize=11)
        axes[1].set_ylabel('Score', fontsize=11)
        axes[1].set_title('Die Model - Per-Class Performance', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x_die)
        axes[1].set_xticklabels(die_classes, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Performance metrics saved to {save_path}")
        plt.close()
    
    def plot_confidence_analysis(self, save_path: str):
        """
        Analyze confidence distributions
        
        Args:
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract confidence data from metrics
        wafer_conf = {
            'mean': self.wafer_metrics['metrics']['avg_confidence'],
            'min': self.wafer_metrics['metrics']['min_confidence'],
            'max': self.wafer_metrics['metrics']['max_confidence']
        }
        
        die_conf = {
            'mean': self.die_metrics['metrics']['avg_confidence'],
            'min': self.die_metrics['metrics']['min_confidence'],
            'max': self.die_metrics['metrics']['max_confidence']
        }
        
        # Wafer confidence bars
        conf_data = ['Mean', 'Min', 'Max']
        wafer_values = [wafer_conf['mean'], wafer_conf['min'], wafer_conf['max']]
        
        axes[0].barh(conf_data, wafer_values, color=['#2E86AB', '#E63946', '#06A77D'])
        axes[0].axvline(0.6, color='red', linestyle='--', linewidth=2, label='Threshold (0.6)')
        axes[0].axvline(0.95, color='green', linestyle='--', linewidth=2, label='Early Exit (0.95)')
        axes[0].set_xlabel('Confidence', fontsize=11)
        axes[0].set_title('Wafer Model - Confidence Statistics', fontsize=12, fontweight='bold')
        axes[0].set_xlim([0, 1])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Die confidence bars
        die_values = [die_conf['mean'], die_conf['min'], die_conf['max']]
        
        axes[1].barh(conf_data, die_values, color=['#2E86AB', '#E63946', '#06A77D'])
        axes[1].axvline(0.6, color='red', linestyle='--', linewidth=2, label='Threshold (0.6)')
        axes[1].axvline(0.95, color='green', linestyle='--', linewidth=2, label='Early Exit (0.95)')
        axes[1].set_xlabel('Confidence', fontsize=11)
        axes[1].set_title('Die Model - Confidence Statistics', fontsize=12, fontweight='bold')
        axes[1].set_xlim([0, 1])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Confidence analysis saved to {save_path}")
        plt.close()
    
    def plot_latency_comparison(self, save_path: str):
        """
        Compare inference latency
        
        Args:
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = ['Wafer Model', 'Die Model']
        latencies = [
            self.wafer_metrics['metrics']['avg_latency_ms'],
            self.die_metrics['metrics']['avg_latency_ms']
        ]
        
        colors = ['#2E86AB', '#A23B72']
        bars = ax.bar(models, latencies, color=colors, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, latency in zip(bars, latencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{latency:.2f} ms',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Target latency line
        ax.axhline(50, color='red', linestyle='--', linewidth=2, label='Target (50 ms)')
        
        ax.set_ylabel('Average Latency (ms)', fontsize=12)
        ax.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Latency comparison saved to {save_path}")
        plt.close()
    
    def generate_summary_table(self, save_path: str):
        """
        Create summary table with key metrics
        
        Args:
            save_path: Path to save CSV
        """
        summary_data = {
            'Model': ['Wafer', 'Die'],
            'Accuracy': [
                self.wafer_metrics['metrics']['accuracy'],
                self.die_metrics['metrics']['accuracy']
            ],
            'Avg Precision': [
                np.mean(list(self.wafer_metrics['metrics']['per_class_precision'].values())),
                np.mean(list(self.die_metrics['metrics']['per_class_precision'].values()))
            ],
            'Avg Recall': [
                np.mean(list(self.wafer_metrics['metrics']['per_class_recall'].values())),
                np.mean(list(self.die_metrics['metrics']['per_class_recall'].values()))
            ],
            'Avg F1-Score': [
                np.mean(list(self.wafer_metrics['metrics']['per_class_f1'].values())),
                np.mean(list(self.die_metrics['metrics']['per_class_f1'].values()))
            ],
            'Avg Confidence': [
                self.wafer_metrics['metrics']['avg_confidence'],
                self.die_metrics['metrics']['avg_confidence']
            ],
            'Latency (ms)': [
                self.wafer_metrics['metrics']['avg_latency_ms'],
                self.die_metrics['metrics']['avg_latency_ms']
            ]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_csv(save_path, index=False, float_format='%.4f')
        
        print(f"\n{'='*60}")
        print("MODEL PERFORMANCE SUMMARY")
        print('='*60)
        print(df.to_string(index=False))
        print('='*60)
        
        print(f"\n✓ Summary table saved to {save_path}")
    
    def create_full_report(self, output_dir: str):
        """
        Generate all visualizations and reports
        
        Args:
            output_dir: Directory to save all outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60 + "\n")
        
        self.load_metrics()
        
        # Generate all plots
        self.plot_training_comparison(str(output_path / 'training_comparison.png'))
        self.plot_performance_metrics(str(output_path / 'performance_metrics.png'))
        self.plot_confidence_analysis(str(output_path / 'confidence_analysis.png'))
        self.plot_latency_comparison(str(output_path / 'latency_comparison.png'))
        
        # Generate summary table
        self.generate_summary_table(str(output_path / 'summary_metrics.csv'))
        
        print("\n" + "="*60)
        print(f"✅ Full report generated in {output_dir}")
        print("="*60)
        
        print("\nGenerated Files:")
        for file in output_path.glob('*'):
            print(f"  - {file.name}")


def main():
    """CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="Visualize and analyze training results"
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='outputs/results',
        help='Directory containing result files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_report',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ResultsVisualizer(args.results_dir)
    
    # Generate full report
    visualizer.create_full_report(args.output_dir)


if __name__ == "__main__":
    main()
