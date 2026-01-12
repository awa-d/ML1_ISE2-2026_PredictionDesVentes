# 📊 MLflow Experiment Tracking Guide

## 🎯 Overview

This project now includes comprehensive **MLflow experiment tracking** to monitor and compare all machine learning experiments with full reproducibility.

## 🚀 What's Being Tracked

### 1. **Hyperparameters** 📝
All model and training configurations:
- `objective`, `metric`, `boosting_type`
- `learning_rate`, `num_leaves`, `feature_fraction`
- `bagging_fraction`, `bagging_freq`, `min_data_in_leaf`
- `num_boost_rounds`, `early_stopping_rounds`
- `weight_perishable`, `weight_normal`
- Random seed for reproducibility

### 2. **Dataset Information** 📦
- Training set size (samples & features)
- Validation set size (samples & features)
- Number of features used
- List of all features
- Target transformation type (log1p)

### 3. **Training Metrics** 📈
Logged at each iteration:
- **train_rmse**: Training RMSE over iterations
- **valid_rmse**: Validation RMSE over iterations
- **best_iteration**: Best iteration from early stopping
- **num_trees**: Total number of trees in final model

### 4. **Final Performance Metrics** 🎯
- **nwrmsle_train**: Normalized Weighted RMSLE on training set
- **nwrmsle_valid**: Normalized Weighted RMSLE on validation set (PRIMARY METRIC)
- **mae_valid_original_scale**: Mean Absolute Error on original scale
- **rmse_valid_original_scale**: Root Mean Squared Error on original scale

### 5. **Feature Importance** 🔍
- Gain-based importance for all features
- Saved as JSON artifact
- Top 10 displayed in console

### 6. **Model Artifacts** 💾
- Full LightGBM model (MLflow format)
- Local model save (.txt format)
- Feature information (JSON)

### 7. **Run Metadata** 🏷️
- **model_type**: LightGBM
- **task**: sales_forecasting
- **dataset**: favorita
- **status**: completed
- **run_name**: Timestamped (e.g., `lgb_run_20260108_143022`)

## 📂 File Structure

```
favorita-sales-forecasting_ENSAE-ISE2-2026/
├── src/
│   └── train.py              # Updated with MLflow tracking
├── mlruns/                   # MLflow tracking directory (auto-created)
│   └── [experiment_id]/
│       └── [run_id]/
│           ├── metrics/      # All logged metrics
│           ├── params/       # All logged parameters
│           ├── artifacts/    # Model & feature importance
│           └── meta.yaml     # Run metadata
└── MLFLOW_GUIDE.md          # This guide
```

## 🎓 Usage

### 1. Install Dependencies
```bash
cd favorita-sales-forecasting_ENSAE-ISE2-2026
pip install -r requirements.txt
```

### 2. Run Training (Tracking Enabled)
```bash
cd src
python train.py
```

**Output includes:**
```
🚀 DÉBUT DE L'EXPÉRIENCE DE MACHINE LEARNING
📊 Experiment: favorita-sales-forecasting
🔖 Run ID: a1b2c3d4e5f6...
================================================================================
...training logs...
================================================================================
✅ EXPÉRIENCE TERMINÉE AVEC SUCCÈS
🎯 Score Final (NWRMSLE): 0.53412
📂 MLflow Tracking URI: ./mlruns
🔗 Run ID: a1b2c3d4e5f6...
================================================================================
```

### 3. View Results in MLflow UI
```bash
cd favorita-sales-forecasting_ENSAE-ISE2-2026
mlflow ui
```

Then open your browser: **http://localhost:5000**

## 🖥️ MLflow UI Features

### **Experiments Table**
- Compare all runs side-by-side
- Sort by any metric (NWRMSLE, RMSE, MAE)
- Filter by parameters or tags
- Search runs by name or ID

### **Run Details Page**
For each run, view:
- **Overview**: Duration, status, timestamps
- **Parameters**: All hyperparameters
- **Metrics**: Interactive charts showing training curves
- **Artifacts**: Download model, feature importance JSON
- **Tags**: Metadata and classification

### **Compare Runs**
- Select multiple runs
- See parameter differences highlighted
- Compare metric charts overlayed
- Export comparison to CSV

## 📊 Example: Analyzing Results

### Find Best Model
1. Open MLflow UI
2. Sort experiments by `nwrmsle_valid` (ascending)
3. Click on best run
4. Review parameters used
5. Download model from artifacts

### Compare Hyperparameters
1. Select 2-3 runs with different `learning_rate`
2. Click "Compare"
3. See which learning rate performed best
4. Check training curves for overfitting

### Track Feature Importance
1. Open any run
2. Go to "Artifacts"
3. Download `feature_importance_gain.json`
4. Analyze which features matter most

## 🔄 Running Multiple Experiments

### Example: Tune Learning Rate
```python
# Modify train.py - LGB_PARAMS section:
for lr in [0.01, 0.05, 0.1]:
    LGB_PARAMS["learning_rate"] = lr
    # ... run training ...
    # Each run logged separately in MLflow
```

### Example: Test Different Features
```python
# Modify preprocessing.py to exclude certain features
# Run train.py multiple times
# Compare feature_info.json and nwrmsle_valid across runs
```

## 📈 Key Metrics Explained

### **NWRMSLE (Primary Metric)**
- **Normalized Weighted Root Mean Squared Logarithmic Error**
- Kaggle competition metric
- Penalizes underestimation
- Weights perishable items 1.25x
- **Lower is better**

### **RMSE (Training Metric)**
- Used during training for early stopping
- Logged at each iteration
- Tracks model convergence

### **MAE & RMSE (Original Scale)**
- Interpretable metrics
- Show error in actual sales units
- Help understand business impact

## 🎯 Best Practices

1. **Run Naming**: Uses timestamps automatically for easy identification
2. **Version Control**: Git commit hash can be added as a tag
3. **Data Versioning**: Log dataset hash/size to track data changes
4. **Reproducibility**: All hyperparameters logged for exact reproduction
5. **Model Registry**: Use MLflow Model Registry for production deployment

## 🆘 Troubleshooting

### MLflow UI won't start
```bash
# Check if port 5000 is in use
netstat -an | findstr "5000"

# Use different port
mlflow ui --port 5001
```

### Can't find mlruns directory
```bash
# Check tracking URI
echo $MLFLOW_TRACKING_URI

# Should be: ./mlruns (relative to project root)
```

### Metrics not logging
- Check MLflow version: `pip show mlflow`
- Ensure within `mlflow.start_run()` context
- Check console for error messages

## 🔗 Advanced Features

### Remote Tracking Server
```python
# train.py - modify MLFLOW_TRACKING_URI
MLFLOW_TRACKING_URI = "http://your-server:5000"
```

### Model Registry
```python
# After training, register model
mlflow.register_model(
    f"runs:/{run_id}/model",
    "favorita-forecaster"
)
```

### Load Model for Inference
```python
import mlflow.lightgbm

# Load by run ID
model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")

# Or load by model name
model = mlflow.lightgbm.load_model("models:/favorita-forecaster/1")
```

## 📚 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [LightGBM + MLflow Integration](https://mlflow.org/docs/latest/python_api/mlflow.lightgbm.html)
- [Experiment Tracking Tutorial](https://mlflow.org/docs/latest/tracking.html)

---

**Happy Experimenting! 🚀**
