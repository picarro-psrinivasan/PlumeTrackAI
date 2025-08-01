#!/usr/bin/env python3
"""
Evaluation metrics for PlumeTrackAI.
Contains functions for calculating model performance metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics for model evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary containing MSE, MAE, RMSE, and R²
    """
    # Calculate basic metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Model Evaluation"):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print(f"\n=== {title} ===")
    print(f"Mean Squared Error: {metrics['MSE']:.4f}")
    print(f"Mean Absolute Error: {metrics['MAE']:.4f}")
    print(f"Root Mean Squared Error: {metrics['RMSE']:.4f}")
    print(f"R-squared: {metrics['R2']:.4f}")


def calculate_wind_prediction_accuracy(wind_speed_true: np.ndarray, 
                                     wind_speed_pred: np.ndarray,
                                     wind_direction_true: np.ndarray = None,
                                     wind_direction_pred: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate accuracy metrics specifically for wind prediction.
    
    Args:
        wind_speed_true: True wind speeds
        wind_speed_pred: Predicted wind speeds
        wind_direction_true: True wind directions (optional)
        wind_direction_pred: Predicted wind directions (optional)
    
    Returns:
        Dictionary of wind-specific metrics
    """
    # Basic speed metrics
    speed_metrics = calculate_regression_metrics(wind_speed_true, wind_speed_pred)
    
    # Direction metrics (if provided)
    direction_metrics = {}
    if wind_direction_true is not None and wind_direction_pred is not None:
        # Calculate circular mean absolute error for directions
        angle_diff = np.abs(wind_direction_true - wind_direction_pred)
        # Handle circular nature (e.g., 359° and 1° are only 2° apart)
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)
        direction_mae = np.mean(angle_diff)
        direction_metrics['Direction_MAE'] = direction_mae
    
    # Combine metrics
    all_metrics = {**speed_metrics, **direction_metrics}
    
    return all_metrics 