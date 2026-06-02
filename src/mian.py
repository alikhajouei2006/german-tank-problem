from simulation import monte_carlo
from statistic import compute_metric
from visualization import plot_bar_comparison
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

def main():
    N_real = 1000
    sample_sizes = [5, 10, 20, 50, 100]
    n_simulations = 1000

    results = monte_carlo(sample_sizes,
                          N_real,
                          n_simulations)
    
    metrics = ['mae', 'var', 'bias', 'mse']
    for metric in metrics:
        metric_res = compute_metric(sample_sizes,
                                    results,
                                    N_real,
                                    None,
                                    metric)
    
        fig = plot_bar_comparison(metric_res,
                            metric)
        save_path = FIGURES_DIR / f"{metric}_comparison.png"

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print("saved")
    
    
if __name__ == "__main__":
    main()
