import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

def plot_bar_comparison(metric_results: dict,
                        label: str
                        ) -> mpl.figure.Figure:
    """
        This method plots a bar chart for a specific estimator
        across all smaple sizes.
    """
    if not isinstance(metric_results, dict):
        raise TypeError(
        f'{label.title()} results must be a nested dict.'
    )
    if not metric_results:
        raise ValueError(
            f'{label.title()} results cannot be empty.')

    fig, ax = plt.subplots(figsize=(10, 6))

    pd.DataFrame(metric_results, index=None).T.plot(kind='bar', width=0.4, ax=ax, zorder=3)

    ax.set_xlabel('Number of samples', font='Times New Roman', fontsize=15)
    ax.set_xticklabels(list(metric_results.keys()), rotation=0, font='Times New Roman', fontsize=12)

    ax.set_title(f'{label.title()} comparison across different sample sizes', font='Times New Roman', fontsize=15)

    ax.set_ylabel(label.title(), font='Times New Roman', fontsize=15)

    ax.grid(
        axis='y',
        linestyle=':',
        linewidth=1,
        color='lightgrey',
        zorder=0
    )

    ax.legend(prop=dict(family='Times New Roman', size=9))

    plt.tight_layout()

    plt.close()
    return fig