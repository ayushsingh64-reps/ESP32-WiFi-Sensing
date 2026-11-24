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
    """
    is_series = isinstance(data, pd.Series)
    if is_series:
        s = data.copy()
    else:
        s = pd.Series(data)

    k = 1.4826  # Scale factor for Gaussian distribution consistency
    rolling_median = s.rolling(window=2 * window_size + 1, min_periods=1, center=True).median()
    
    abs_diff = (s - rolling_median).abs()
    mad = abs_diff.rolling(window=2 * window_size + 1, min_periods=1, center=True).median()
    threshold = n_sigma * k * mad
    
    outlier_mask = abs_diff > threshold
    outlier_indices = s.index[outlier_mask].tolist()
    
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
