import kfp
from kfp import dsl
from kfp.dsl import component

# =============================================================================
# Component 1: Load Data
# =============================================================================

@component(
    base_image='python:3.9',
    packages_to_install=['pandas==1.3.5']
)
def load_data_op(
    url: str,
    dataset_path: dsl.Output[dsl.Dataset],
):
    """Loads the wine quality dataset from a URL."""
    import pandas as pd

    print(f"Loading data from {url}...")
    df = pd.read_csv(url, sep=';')
    df.to_csv(dataset_path.path, index=False)
    print(f"Data saved to {dataset_path.path}")

# =============================================================================
# Component 2: Train Model
# =============================================================================

@component(
    base_image='python:3.9',
    packages_to_install=['pandas==1.3.5', 'scikit-learn==1.0.2', 'joblib==1.1.0']
)
def train_model_op(
    dataset_path: dsl.Input[dsl.Dataset],
    model_path: dsl.Output[dsl.Model],
    test_data_path: dsl.Output[dsl.Dataset],
):
    """Trains a RandomForestRegressor and saves the model."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    print("Loading data for training...")
    df = pd.read_csv(dataset_path.path)
    X = df.drop('quality', axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path.path)
    print(f"Model saved to {model_path.path}")

    # Save test data for evaluation
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(test_data_path.path, index=False)
    print(f"Test data saved to {test_data_path.path}")

# =============================================================================
# Component 3: Evaluate Model
# =============================================================================

@component(
    base_image='python:3.9',
    packages_to_install=['pandas==1.3.5', 'scikit-learn==1.0.2', 'joblib==1.1.0', 'numpy==1.21.6']
)
def evaluate_model_op(
    model_path: dsl.Input[dsl.Model],
    test_data_path: dsl.Input[dsl.Dataset],
    metrics_path: dsl.Output[dsl.Metrics],
):
    """Evaluates the model and saves the RMSE metric."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error
    import joblib
    import json

    print("Loading model and test data for evaluation...")
    model = joblib.load(model_path.path)
    test_df = pd.read_csv(test_data_path.path)

    X_test = test_df.drop('quality', axis=1)
    y_test = test_df['quality']

    print("Evaluating model...")
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Root Mean Squared Error: {rmse:.4f}")

    # Save metrics as a JSON file for Kubeflow UI
    metrics = {
        'metrics': [{
            'name': 'rmse',
            'numberValue': rmse,
            'format': "RAW",
        }]
    }
    with open(metrics_path.path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_path.path}")

# =============================================================================
# Pipeline Definition
# =============================================================================

@dsl.pipeline(
    name='Wine Quality Prediction Pipeline',
    description='A pipeline that trains and evaluates a model to predict wine quality.'
)
def wine_quality_pipeline(
    data_url: str = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
):
    """Defines the Kubeflow pipeline structure."""
    # Define pipeline flow
    load_task = load_data_op(url=data_url)
    
    train_task = train_model_op(
        dataset_path=load_task.outputs['dataset_path']
    )
    
    evaluate_model_op(
        model_path=train_task.outputs['model_path'],
        test_data_path=train_task.outputs['test_data_path']
    )

# =============================================================================
# Compile the pipeline
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'kubeflow':
        if len(sys.argv) > 2 and sys.argv[2] == 'run':
            # Run on Kubeflow
            print("Running pipeline on Kubeflow...")
            print("Note: You need to configure your Kubeflow endpoint first.")
            print("Example commands:")
            print("1. For Google Cloud: gcloud auth login && gcloud config set project YOUR_PROJECT")
            print("2. For local Kubeflow: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80")
            print("3. Then run: poetry run python wine_pipeline.py kubeflow run --endpoint http://localhost:8080")
            
            # You can uncomment and modify these lines to actually run:
            # client = kfp.Client(host='YOUR_KUBEFLOW_ENDPOINT')
            # run = client.create_run_from_pipeline_func(
            #     wine_quality_pipeline,
            #     arguments={'data_url': 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'}
            # )
            # print(f"Pipeline run submitted: {run.run_id}")
        else:
            print("Usage: python wine_pipeline.py kubeflow run")
    else:
        # Compile to YAML
        print("Compiling pipeline to wine_pipeline.yaml...")
        kfp.compiler.Compiler().compile(
            pipeline_func=wine_quality_pipeline,
            package_path='wine_pipeline.yaml'
        )
        print("Pipeline compiled successfully.")
        print("\nTo run the pipeline:")
        print("1. Upload wine_pipeline.yaml to your Kubeflow UI")
        print("2. Or use: poetry run python wine_pipeline.py kubeflow run")
