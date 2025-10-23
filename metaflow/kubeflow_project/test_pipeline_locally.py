#!/usr/bin/env python3
"""
Test the wine pipeline components locally without Kubeflow
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import json

def test_pipeline_locally():
    """Run the pipeline components locally for testing"""
    
    # Step 1: Load Data
    print("Step 1: Loading data...")
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df = pd.read_csv(url, sep=';')
    print(f"Loaded {len(df)} rows of data")
    
    # Step 2: Train Model
    print("\nStep 2: Training model...")
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'wine_quality_model.joblib')
    print("Model saved to wine_quality_model.joblib")
    
    # Step 3: Evaluate Model
    print("\nStep 3: Evaluating model...")
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2_score = model.score(X_test, y_test)
    
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2_score:.4f}")
    
    # Save metrics
    metrics = {
        'rmse': float(rmse),
        'r2_score': float(r2_score),
        'n_estimators': 100,
        'test_size': 0.2
    }
    
    with open('wine_quality_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to wine_quality_metrics.json")
    
    print("\n✅ Pipeline completed successfully!")

if __name__ == '__main__':
    test_pipeline_locally()