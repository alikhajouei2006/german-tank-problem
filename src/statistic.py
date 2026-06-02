import numpy as np

def compute_metric(sample_sizes : list[int],
                   results : dict,
                   N_real : int = 1000,
                   methods: list[str] | None = None,
                   kind : str ='mae'
                   ) -> dict[int, dict]:
    """
        Compute specified metric for each estimator 
        across all sample sizes.

        Returns:
            nested dictionary containing specified metric results
            for each sample size.
    """
    _validate_inputs(sample_sizes,
                    results,
                    N_real,
                    kind)
    if methods is None:
        methods = ["mle", "mvue", "mom"]

    metric_results = {
        size: {
            method: None
            for method in methods
        }
        for size in sample_sizes
    }   
    
    kind = kind.lower()

    for size in sample_sizes:
        for method in methods:
            estimates = np.asarray(results[size][method])

            if kind in ['mae', 'mean_absolute_error']:
                error = np.abs(estimates - N_real)
                mae = np.mean(error)
                metric_results[size][method] = mae
            elif kind in ['var', 'variance']:
                if len(estimates) < 2:
                    raise ValueError("At least two simulation results are required for variance computation.")
                variance = np.var(estimates, ddof=1)
                metric_results[size][method] = variance
            elif kind == 'bias':
                mean_estimate = np.mean(estimates)
                bias = mean_estimate - N_real
                metric_results[size][method] = bias
            elif kind in ['mse', 'mean_squared_error']:
                mean = np.mean(estimates)
                bias = mean - N_real
                if len(estimates) < 2:
                    raise ValueError("At least two simulation results are required for variance computation.")
                variance = np.var(estimates, ddof=1)
                mse = variance + bias ** 2
                metric_results[size][method] = mse
                     
    return metric_results

def _validate_inputs(sample_sizes : list[int],
                    results : dict, 
                    N_real : int,
                    kind : str) -> None:
    """
        This method has been written to validate
        statistical methods and prevent code repetition.
    """
    if not isinstance(kind, str):
        raise TypeError(
            'Kind must be a string.'
        )
    if kind.lower() not in ['mae', 'var', 'bias', 'mse',
                            'mean_absolute_error', 'variance',
                            'mean_squared_error']:
        raise ValueError(
            f'''The specified kind: {kind} is invalid.
            please, specify a valid kind: 
            ['mae', 'var', 'bias', 'mse',
             'mean_absolute_error', 'variance',
             'mean_squared_error']. '''
        )
    if not results or not sample_sizes:
        raise ValueError(
            "results dictionary or sample sizes cannot be empty."
        )
    if N_real <= 0:
        raise ValueError(
            "N_real must be positive"
        )
    if any(size > N_real for size in sample_sizes):
        raise ValueError(
        "sample size cannot be larger than population size."
    )
    if any(size <= 0 for size in sample_sizes):
        raise ValueError(
        "sample size must be positive."
    )
    
