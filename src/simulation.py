import numpy as np
from estimators import mle, mvue, mom


def monte_carlo(sample_sizes : list[int], 
                N_real : int = 1000, 
                n_simulations : int = 1000, 
                seed : int = 42 ) -> dict:
    """
        execution of Monte Carlo simulation for German Tank problem
    """
    if n_simulations <= 0:
        raise ValueError(
            "n_simulations must be positive"
        )
    if any(size > N_real for size in sample_sizes):
        raise ValueError(
        "sample size cannot be larger than population size"
        )
    if any(size <= 0 for size in sample_sizes):
        raise ValueError(
        "sample size must be positive"
        )
    
    results = {
    size: {
        "mle": [],
        "mvue": [],
        "mom": []
        }
        for size in sample_sizes
    }
    rng = np.random.default_rng(seed)

    for size in sample_sizes:
        for _ in range(n_simulations):

            sample = rng.choice(
                                np.arange(1, N_real + 1),
                                size=size,
                                replace=False
                                )

            results[size]['mle'].append(mle(sample))
            results[size]['mvue'].append(mvue(sample))
            results[size]['mom'].append(mom(sample))

    return results