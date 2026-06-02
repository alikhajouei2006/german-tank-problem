import numpy as np

def mle(sample : np.ndarray) -> np.int64:
    """
        Maximum likelihood estimator for German Tank problem
    """
    if sample.size == 0:
        raise ValueError("sample cannot be empty")
    
    return np.max(sample)

def mvue(sample : np.ndarray) -> float:
    """
        Minimum-variance unbiased estimator for German Tank problem
    """
    if sample.size == 0:
        raise ValueError("sample cannot be empty")
    
    maximum = np.max(sample)
    k = len(sample)
    return maximum * (1 + 1 / k) - 1

def mom(sample : np.ndarray) -> float:
    """
        Method of Moments estimator for German Tank problem
    """
    if sample.size == 0:    
        raise ValueError("sample cannot be empty")
    
    mean = np.mean(sample)
    return 2 * mean - 1