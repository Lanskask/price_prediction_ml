
from metaflow import FlowSpec, step, Parameter, card
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import joblib
import json
from datetime import datetime

class WineQualityFlow(FlowSpec):
    """
    A comprehensive Metaflow pipeline that trains and evaluates a model to predict wine quality.
    
    This pipeline:
    1. Loads wine quality data from UCI ML repository
    2. Performs data validation and preprocessing
    3. Trains a RandomForestRegressor model
    4. Evaluates the model with multiple metrics
    5. Saves the model and metrics for future use
    6. Provides detailed logging and monitoring
    """
    
    # Parameters allow runtime configuration without code changes
    data_url = Parameter(
        'data_url',
        help='URL of the wine quality dataset',
        default='http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    )
    
    test_size = Parameter(
        'test_size',
        help='Fraction of data to use for testing',
        default=0.2
    )
    
    n_estimators = Parameter(
        'n_estimators',
        help='Number of trees in the RandomForest',
        default=100
    )
    
    random_state = Parameter(
        'random_state',
        help='Random seed for reproducibility',
        default=42
    )

    @step
    def start(self):
        """
        Load and validate the wine quality dataset.
        
        Why this step:
        - Data loading is a critical first step that can fail
        - Validation ensures data quality before processing
        - Separating concerns makes debugging easier
        """
        print(f"Loading wine quality data from {self.data_url}...")
        
        try:
            # Load the dataset
            self.df = pd.read_csv(self.data_url, sep=';')
            print(f"✅ Successfully loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Data validation
            if 'quality' not in self.df.columns:
                raise ValueError("Missing 'quality' column in dataset")
            
            if len(self.df) == 0:
                raise ValueError("Dataset is empty")
            
            # Store basic statistics for monitoring
            self.data_stats = {
                'n_samples': int(len(self.df)),
                'n_features': int(len(self.df.columns) - 1),  # Excluding target
                'quality_range': (int(self.df['quality'].min()), int(self.df['quality'].max())),
                'missing_values': int(self.df.isnull().sum().sum())
            }
            
            print(f"📊 Data Statistics:")
            print(f"   - Samples: {self.data_stats['n_samples']}")
            print(f"   - Features: {self.data_stats['n_features']}")
            print(f"   - Quality range: {self.data_stats['quality_range']}")
            print(f"   - Missing values: {self.data_stats['missing_values']}")
            
            # Prepare features and target
            self.X = self.df.drop('quality', axis=1)
            self.y = self.df['quality']
            
            print("✅ Data preparation complete")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
        
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train a RandomForestRegressor model with the prepared data.
        
        Why this step:
        - Model training is computationally expensive and should be isolated
        - Allows for easy hyperparameter tuning via Parameters
        - Enables model versioning and comparison
        """
        print("🌲 Training RandomForestRegressor model...")
        
        try:
            # Split the data
            print(f"📊 Splitting data (test_size={self.test_size})...")
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            
            print(f"   - Training samples: {len(X_train)}")
            print(f"   - Test samples: {len(X_test)}")
            
            # Train the model
            print(f"🌲 Training RandomForest with {self.n_estimators} trees...")
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators, 
                random_state=self.random_state,
                n_jobs=-1  # Use all available cores
            )
            
            self.model.fit(X_train, y_train)
            
            # Store test data for evaluation
            self.X_test = X_test
            self.y_test = y_test
            
            # Store training metadata
            self.training_metadata = {
                'n_estimators': int(self.n_estimators),
                'test_size': float(self.test_size),
                'random_state': int(self.random_state),
                'training_samples': int(len(X_train)),
                'test_samples': int(len(X_test)),
                'feature_importance': {col: float(imp) for col, imp in zip(self.X.columns, self.model.feature_importances_)}
            }
            
            print("✅ Model training complete")
            print(f"📈 Top 5 most important features:")
            top_features = sorted(self.training_metadata['feature_importance'].items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in top_features:
                print(f"   - {feature}: {importance:.4f}")
            
        except Exception as e:
            print(f"❌ Error training model: {str(e)}")
            raise
        
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        """
        Evaluate the trained model using multiple metrics.
        
        Why this step:
        - Comprehensive evaluation provides better model understanding
        - Multiple metrics give different perspectives on performance
        - Enables model comparison and selection
        """
        print("📊 Evaluating model performance...")
        
        try:
            # Make predictions
            predictions = self.model.predict(self.X_test)
            
            # Calculate multiple metrics
            self.rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
            self.mae = mean_absolute_error(self.y_test, predictions)
            self.r2 = r2_score(self.y_test, predictions)
            
            # Additional metrics for comprehensive evaluation
            self.mse = mean_squared_error(self.y_test, predictions)
            self.residuals = self.y_test - predictions
            
            # Store comprehensive metrics
            self.metrics = {
                'rmse': float(self.rmse),
                'mae': float(self.mae),
                'r2_score': float(self.r2),
                'mse': float(self.mse),
                'mean_residual': float(np.mean(self.residuals)),
                'std_residual': float(np.std(self.residuals)),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            print("✅ Model evaluation complete")
            print(f"📈 Performance Metrics:")
            print(f"   - RMSE: {self.rmse:.4f}")
            print(f"   - MAE: {self.mae:.4f}")
            print(f"   - R² Score: {self.r2:.4f}")
            print(f"   - MSE: {self.mse:.4f}")
            
            # Model quality assessment
            if self.r2 > 0.7:
                quality = "Excellent"
            elif self.r2 > 0.5:
                quality = "Good"
            elif self.r2 > 0.3:
                quality = "Fair"
            else:
                quality = "Poor"
            
            print(f"🎯 Model Quality: {quality} (R² = {self.r2:.4f})")
            
        except Exception as e:
            print(f"❌ Error evaluating model: {str(e)}")
            raise
        
        self.next(self.save_model)

    @step
    def save_model(self):
        """
        Save the trained model and metrics for future use.
        
        Why this step:
        - Model persistence enables reuse without retraining
        - Metrics storage enables model comparison over time
        - Artifacts can be accessed by other flows or external systems
        """
        print("💾 Saving model and artifacts...")
        
        try:
            # Save the model
            model_filename = f"wine_quality_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(self.model, model_filename)
            self.model_path = model_filename
            
            # Save metrics
            metrics_filename = f"wine_quality_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metrics_filename, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            self.metrics_path = metrics_filename
            
            # Save comprehensive results
            results = {
                'model_metadata': self.training_metadata,
                'performance_metrics': self.metrics,
                'data_statistics': self.data_stats,
                'pipeline_parameters': {
                    'data_url': self.data_url,
                    'test_size': self.test_size,
                    'n_estimators': self.n_estimators,
                    'random_state': self.random_state
                }
            }
            
            results_filename = f"wine_quality_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_filename, 'w') as f:
                json.dump(results, f, indent=2)
            self.results_path = results_filename
            
            print("✅ Artifacts saved successfully")
            print(f"   - Model: {self.model_path}")
            print(f"   - Metrics: {self.metrics_path}")
            print(f"   - Results: {self.results_path}")
            
        except Exception as e:
            print(f"❌ Error saving artifacts: {str(e)}")
            raise
        
        self.next(self.end)

    @step
    def end(self):
        """
        Final step - summarize the pipeline execution.
        
        Why this step:
        - Provides a clear completion summary
        - Can trigger notifications or downstream processes
        - Enables monitoring and alerting
        """
        print("🎉 WineQualityFlow completed successfully!")
        print(f"📊 Final Summary:")
        print(f"   - Model Performance: RMSE={self.rmse:.4f}, R²={self.r2:.4f}")
        print(f"   - Model Quality: {'Excellent' if self.r2 > 0.7 else 'Good' if self.r2 > 0.5 else 'Fair' if self.r2 > 0.3 else 'Poor'}")
        print(f"   - Artifacts saved: {self.model_path}")
        print(f"   - Pipeline completed at: {datetime.now().isoformat()}")

if __name__ == '__main__':
    WineQualityFlow()
