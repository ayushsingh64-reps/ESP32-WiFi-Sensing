"""
Pure Python/NumPy/Pandas implementation of the Hampel filter.
Drop-in replacement for the C-extension `hampel` library.
Compatible with pandas Series and 1D numpy arrays.
"""

from typing import Union, List
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Result:
    filtered_data: Union[pd.Series, np.ndarray]
    outlier_indices: List[int]
    medians: Union[pd.Series, np.ndarray]
    thresholds: Union[pd.Series, np.ndarray]


def hampel(data: Union[pd.Series, np.ndarray, list], window_size: int = 5, n_sigma: float = 3.0) -> Result:
    """
    Apply Hampel filter to detect and replace outliers in a sequence.
    
    Parameters:
    -----------
    data : pd.Series or np.ndarray or list
        Input 1D time series.
    window_size : int, default=5
        Size of the moving window (one-sided radius or rolling window size).
    n_sigma : float, default=3.0
        Threshold multiplier for MAD (Median Absolute Deviation).
        
    Returns:
    --------
    Result:
        Dataclass containing:
        - filtered_data: series/array with outliers replaced by rolling medians
        - outlier_indices: list of indices of detected outliers
        - medians: rolling median sequence
        - thresholds: rolling threshold values (n_sigma * 1.4826 * MAD)
    """
    is_series = isinstance(data, pd.Series)
    if is_series:
        s = data.copy()
    else:
        s = pd.Series(data)

    # Compute rolling median
    # window = 2 * window_size + 1 (centered window) or min_periods=1
    k = 1.4826  # Scale factor for Gaussian distribution consistency
    rolling_median = s.rolling(window=2 * window_size + 1, min_periods=1, center=True).median()
    
    # Absolute difference from median
    abs_diff = (s - rolling_median).abs()
    
    # Median Absolute Deviation (MAD)
    mad = abs_diff.rolling(window=2 * window_size + 1, min_periods=1, center=True).median()
    threshold = n_sigma * k * mad
    
    # Identify outliers
    outlier_mask = abs_diff > threshold
    outlier_indices = s.index[outlier_mask].tolist()
    
    # Replace outliers with median
    filtered = s.copy()
    filtered[outlier_mask] = rolling_median[outlier_mask]
    
    if not is_series:
        filtered_ret = filtered.to_numpy()
        medians_ret = rolling_median.to_numpy()
        threshold_ret = threshold.to_numpy()
    else:
        filtered_ret = filtered
        medians_ret = rolling_median
        threshold_ret = threshold

    return Result(
        filtered_data=filtered_ret,
        outlier_indices=outlier_indices,
        medians=medians_ret,
        thresholds=threshold_ret,
    )
