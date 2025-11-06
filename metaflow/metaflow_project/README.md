# Wine Quality Prediction with Metaflow

A comprehensive machine learning pipeline built with Metaflow that predicts wine quality using the UCI Wine Quality dataset.

## Features

- **Data Validation**: Comprehensive data quality checks and statistics
- **Model Training**: RandomForestRegressor with configurable parameters
- **Model Evaluation**: Multiple metrics (RMSE, MAE, R², MSE)
- **Artifact Persistence**: Saves models, metrics, and comprehensive results
- **Parameter Tuning**: Runtime configuration via Metaflow Parameters
- **Monitoring**: Detailed logging and progress tracking
- **Reproducibility**: Fixed random seeds and versioned artifacts

## Installation

```bash
# Install dependencies
poetry install

# Or with pip
pip install metaflow pandas scikit-learn numpy joblib
```

## Usage

### Basic Run
```bash
poetry run python wine_flow.py run
```

### With Custom Parameters
```bash
# Custom test size and number of trees
poetry run python wine_flow.py run --test_size 0.3 --n_estimators 200

# Custom data source
poetry run python wine_flow.py run --data_url "path/to/your/wine_data.csv"
```

### View Run History
```bash
# List all runs
poetry run python wine_flow.py runs

# View specific run details
poetry run python wine_flow.py runs show RUN_ID
```

### Resume Failed Runs
```bash
# Resume from a specific step
poetry run python wine_flow.py resume RUN_ID
```

## Pipeline Steps

### 1. **start** - Data Loading & Validation
- Downloads wine quality dataset from UCI ML repository
- Validates data integrity (missing values, required columns)
- Calculates and stores data statistics
- Prepares features and target variables

### 2. **train_model** - Model Training
- Splits data into training and testing sets
- Trains RandomForestRegressor with configurable parameters
- Calculates feature importance rankings
- Stores training metadata and model artifacts

### 3. **evaluate_model** - Model Evaluation
- Makes predictions on test set
- Calculates comprehensive metrics (RMSE, MAE, R², MSE)
- Performs residual analysis
- Provides model quality assessment

### 4. **save_model** - Artifact Persistence
- Saves trained model as joblib file
- Exports metrics as JSON
- Creates comprehensive results summary
- Timestamps all artifacts for versioning

### 5. **end** - Pipeline Summary
- Displays final performance summary
- Shows model quality assessment
- Lists saved artifacts

## Output Artifacts

Each run generates timestamped files:

- **`wine_quality_model_TIMESTAMP.joblib`** - Trained RandomForest model
- **`wine_quality_metrics_TIMESTAMP.json`** - Performance metrics
- **`wine_quality_results_TIMESTAMP.json`** - Comprehensive results including:
  - Model metadata (hyperparameters, feature importance)
  - Performance metrics (RMSE, MAE, R², etc.)
  - Data statistics (samples, features, quality range)
  - Pipeline parameters (data URL, test size, etc.)

## Configuration Parameters

| Parameter | Description | Default | Type |
|-----------|-------------|---------|------|
| `data_url` | URL/path to wine quality dataset | UCI ML repository URL | String |
| `test_size` | Fraction of data for testing | 0.2 | Float |
| `n_estimators` | Number of trees in RandomForest | 100 | Integer |
| `random_state` | Random seed for reproducibility | 42 | Integer |

## Model Performance

Typical performance on the wine quality dataset:
- **RMSE**: ~0.55 (lower is better)
- **R² Score**: ~0.54 (higher is better, max 1.0)
- **MAE**: ~0.42 (lower is better)

## Key Features Explained

### Why Metaflow?
- **Reproducibility**: Automatic versioning and artifact tracking
- **Scalability**: Easy deployment to cloud platforms
- **Monitoring**: Built-in logging and progress tracking
- **Resume**: Can resume failed runs from any step
- **Parameters**: Runtime configuration without code changes

### Why These Steps?
- **Separation of Concerns**: Each step has a single responsibility
- **Error Handling**: Isolated steps make debugging easier
- **Parallelization**: Steps can be parallelized in cloud deployments
- **Monitoring**: Each step can be monitored independently

### Why These Metrics?
- **RMSE**: Penalizes large errors more heavily
- **MAE**: Average absolute error, easier to interpret
- **R²**: Proportion of variance explained by the model
- **MSE**: Mean squared error, used for optimization

## Advanced Usage

### Custom Data Sources
```bash
# Use local CSV file
poetry run python wine_flow.py run --data_url "data/wine_quality.csv"

# Use different dataset
poetry run python wine_flow.py run --data_url "https://example.com/wine_data.csv"
```

### Hyperparameter Tuning
```bash
# Test different configurations
poetry run python wine_flow.py run --n_estimators 50 --test_size 0.1
poetry run python wine_flow.py run --n_estimators 200 --test_size 0.3
```

### Production Deployment
```bash
# Deploy to AWS Batch
poetry run python wine_flow.py run --with aws_batch

# Deploy to Kubernetes
poetry run python wine_flow.py run --with kubernetes
```

## Troubleshooting

### Common Issues
1. **Data Loading Errors**: Check URL accessibility and data format
2. **Memory Issues**: Reduce `n_estimators` or use smaller test size
3. **JSON Serialization**: Fixed by converting NumPy types to Python types

### Debug Mode
```bash
# Run with debug output
poetry run python wine_flow.py run --max-workers 1
```

## Next Steps

- Experiment with different algorithms (XGBoost, Neural Networks)
- Add feature engineering steps
- Implement cross-validation
- Add model comparison capabilities
- Deploy to cloud platforms for production use