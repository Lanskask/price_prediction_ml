
from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

class WineQualityFlow(FlowSpec):
    """
    A Metaflow pipeline that trains a model to predict wine quality.
    """
    
    DATA_URL = Parameter(
        'url',
        help='URL of the wine quality dataset',
        default='http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    )

    @step
    def start(self):
        """
        Load the data.
        """
        print("Loading data...")
        self.df = pd.read_csv(self.DATA_URL, sep=';')
        self.X = self.df.drop('quality', axis=1)
        self.y = self.df['quality']
        print(f"Data loaded with {len(self.df)} rows.")
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train a RandomForestRegressor model.
        """
        print("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        self.X_test = X_test
        self.y_test = y_test
        print("Model training complete.")
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        """
        Evaluate the model and print the RMSE.
        """
        print("Evaluating model...")
        predictions = self.model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        self.rmse = rmse
        print(f"Model evaluation complete. Root Mean Squared Error: {self.rmse:.4f}")
        self.next(self.end)

    @step
    def end(self):
        """
        End of the pipeline.
        """
        print("WineQualityFlow finished successfully!")

if __name__ == '__main__':
    WineQualityFlow()
