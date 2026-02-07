# 🚀 Quick Start Guide - IESA DeepTech Hackathon 2026

**Get your defect detection system running in 10 minutes!**

---

## ⚡ Fast Track (Google Colab)

### Step 1: Upload Dataset to Google Drive

```
1. Organize dataset as: Dataset/{train,val,test}/{class_folders}/
2. Compress as Dataset.zip
3. Upload to Google Drive: MyDrive/Dataset.zip
```

### Step 2: Open Colab Notebook

```
1. Go to: https://colab.research.google.com
2. File → Upload Notebook
3. Upload: IESA_QuickStart.ipynb
4. Runtime → Change runtime type → GPU
```

### Step 3: Run All Cells

```
Shift + Enter through all cells (or Runtime → Run All)
```

**That's it!** Training will complete in ~2-3 hours.

---

## 📊 What You'll Get

After training completes:

```
outputs/
├── models/
│   ├── wafer_best.h5           ✅ Trained Keras model
│   ├── wafer_model.onnx        ✅ Edge-ready ONNX
│   ├── die_best.h5
│   └── die_model.onnx
├── results/
│   ├── wafer_metrics.json      📈 Performance metrics
│   ├── confusion_matrix.png    📊 Visualizations
│   ├── confidence_dist.png
│   └── training_curves.png
└── logs/
    └── tensorboard/            📉 TensorBoard logs
```

---

## 🧪 Quick Test

After training, test your model:

```python
from iesa_defect_detection_pipeline import InferencePipeline

# Load models
pipeline = InferencePipeline(
    wafer_model_path="outputs/models/wafer_best.h5",
    die_model_path="outputs/models/die_best.h5"
)

# Predict
result = pipeline.predict("test_image.jpg")
print(f"Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
```

---

## 🔧 Customize Hyperparameters

Edit `config.yaml`:

```yaml
training:
  batch_size: 32      # Reduce if GPU memory issues
  epochs: 50          # Increase for better convergence
  initial_lr: 0.001   # Lower for stable training
```

Then run:
```bash
python iesa_defect_detection_pipeline.py
```

---

## 📱 Deploy to Edge Device

### Convert to TFLite (INT8):

```bash
python convert_to_tflite.py \
  --model outputs/models/wafer_best.h5 \
  --output wafer_int8.tflite \
  --format int8
```

### Benchmark Performance:

```bash
python benchmark_edge_model.py \
  --model wafer_int8.tflite \
  --test-dir Dataset/test \
  --iterations 1000
```

**Expected Results:**
- Latency: ~15ms (CPU)
- Model Size: ~2.3 MB
- Accuracy: ~90%

---

## 📈 Analyze Results

Generate publication-ready plots:

```bash
python visualize_results.py \
  --results-dir outputs/results \
  --output-dir analysis_report
```

**Outputs:**
- Training comparison charts
- Per-class performance metrics
- Confidence distribution analysis
- Latency benchmarks

---

## 🐛 Troubleshooting

### Issue: "Out of Memory" during training

**Solution:**
```yaml
# config.yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: Dataset not found

**Solution:**
```python
# Check path in iesa_defect_detection_pipeline.py (line 150)
DATASET_ROOT = "/content/Dataset"  # Verify this matches your setup
```

### Issue: Low accuracy

**Solutions:**
1. Increase epochs: `epochs: 100` in `config.yaml`
2. Add more training data
3. Adjust class weights (auto-computed by default)
4. Check data quality (remove corrupted images)

---

## 📞 Support

**Documentation:** See `README.md` for detailed explanations  
**Issues:** Create GitHub issue with error logs  
**Email:** your.email@example.com

---

## 🎯 Next Steps

1. ✅ Train base model (you're here!)
2. 🔄 Analyze results & iterate
3. 🚀 Deploy to NXP eIQ edge device
4. 📊 Create hackathon presentation
5. 🏆 Submit to IESA DeepTech 2026

---

**Good luck! 🚀**
