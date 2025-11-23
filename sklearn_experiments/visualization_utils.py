
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from numpy.typing import NDArray

def plot_scaling_with_seaborn(original_data: NDArray[np.float64], scaled_data: NDArray[np.float64], num_features: Optional[int] = 3) -> None:
    """
    Compares original vs. scaled data using Seaborn by first transforming
    the data into a tidy format.

    Args:
        original_data: The dataset before scaling.
        scaled_data: The dataset after scaling.
        num_features: The number of features to plot.
    """
    num_to_plot: int = num_features if num_features is not None else 3
    # Ensure we don't plot more features than available
    if num_to_plot > original_data.shape[1]:
        num_to_plot = original_data.shape[1]

    # Create DataFrames for the first few features
    df_orig: pd.DataFrame = pd.DataFrame(original_data[:, :num_to_plot], columns=[f'Feature {i+1}' for i in range(num_to_plot)])
    df_scaled: pd.DataFrame = pd.DataFrame(scaled_data[:, :num_to_plot], columns=[f'Feature {i+1}' for i in range(num_to_plot)])

    # Add a column to identify the data type
    df_orig['Data Type'] = 'Original'
    df_scaled['Data Type'] = 'Scaled'

    # Combine into a single DataFrame
    combined_df: pd.DataFrame = pd.concat([df_orig, df_scaled])

    # "Melt" the DataFrame from wide to long format
    melted_df: pd.DataFrame = combined_df.melt(id_vars='Data Type', var_name='Feature', value_name='Value')

    # Create the plot
    g: sns.FacetGrid = sns.displot(
        data=melted_df,
        x='Value',
        col='Feature',
        hue='Data Type',
        kind='hist',
        facet_kws={'sharey': False}
    )
    g.figure.suptitle('Original vs. Scaled Data Comparison (Seaborn)', y=1.03)
    plt.show()


def plot_correlation_heatmap(data: NDArray[np.float64]) -> None:
    """
    Calculates and plots the correlation matrix of a dataset as a heatmap.

    Args:
        data: A 2D NumPy array (or pandas DataFrame) where rows are samples
              and columns are features.
    """
    # Convert to a pandas DataFrame for the .corr() method
    df = pd.DataFrame(data)

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Draw the heatmap using Seaborn
    sns.heatmap(corr_matrix, cmap='viridis', annot=False) # annot=True is slow for 100x100

    plt.title('Feature Correlation Matrix')
    plt.show()