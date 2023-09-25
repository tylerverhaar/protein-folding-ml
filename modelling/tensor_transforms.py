import numpy as np
from scipy.stats import boxcox

def min_max_scaling(x: np.ndarray) -> np.ndarray:
    """
    Apply Min-Max Scaling to a NumPy array.

    Parameters:
        x (np.ndarray): Input array to be scaled.

    Returns:
        np.ndarray: Scaled array.
    """
    x_min = np.min(x)
    x_max = np.max(x)
    if x_min == x_max:
        x_max = x_min + 1  # To avoid division by zero
    return (x - x_min) / (x_max - x_min)

def log_transform(x: np.ndarray) -> np.ndarray:
    """
    Apply Logarithmic Transform to a NumPy array.

    Parameters:
        x (np.ndarray): Input array to be transformed.

    Returns:
        np.ndarray: Transformed array.
    """
    if np.any(x <= 0):
        raise ValueError("Log transform requires positive values.")
    return np.log(x)

def box_cox_transform(x: np.ndarray) -> np.ndarray:
    """
    Apply Box-Cox Transform to a NumPy array.

    Parameters:
        x (np.ndarray): Input array to be transformed.

    Returns:
        np.ndarray: Transformed array.
    """
    if np.any(x <= 0):
        raise ValueError("Box-Cox transform requires positive values.")
    return boxcox(x)[0]

def power_transform(x: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    Apply Power Transform to a NumPy array.

    Parameters:
        x (np.ndarray): Input array to be transformed.
        power (float): Power value for the transformation.

    Returns:
        np.ndarray: Transformed array.
    """
    return np.power(x, power)

def standardization(x: np.ndarray) -> np.ndarray:
    """
    Apply Standardization (Z-score normalization) to a NumPy array.

    Parameters:
        x (np.ndarray): Input array to be standardized.

    Returns:
        np.ndarray: Standardized array.
    """
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        std = 1  # To avoid division by zero
    return (x - mean) / std
