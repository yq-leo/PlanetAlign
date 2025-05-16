from typing import Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def plot_loss_curve(loss_logs: dict,
                    save_path: Optional[Union[str, Path]] = None,
                    title: str = 'Loss vs Epochs',
                    xlabel: str = 'Epochs',
                    ylabel: str = 'Loss'):
    """
    Plot the loss curve along training (optimization).

    Parameters
    ----------
    loss_logs : list[float]
        List of loss values for each epoch.
    save_path : str or pathlib.Path, optional
        Path to save the plot. If None, the plot will not be saved. Default is None.
    title : str, optional
        Title of the plot. Default is 'Training Curve'.
    xlabel : str, optional
        Label for the x-axis. Default is 'Epochs'.
    ylabel : str, optional
        Label for the y-axis. Default is 'Loss'.

    Returns
    -------
    None
    """

    plt.figure(figsize=(8, 6))

    for key, values in loss_logs.items():
        plt.plot(range(1, len(values) + 1), values, label=key, linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Calculate y-limits with margin
    all_losses = [value for values in loss_logs.values() for value in values]
    ymin, ymax = min(all_losses), max(all_losses)
    margin = (ymax - ymin) * 0.05
    ymin = max(0, ymin - margin)  # Ensure ymin is non-negative
    ymax += margin
    plt.ylim(ymin, ymax)

    max_epochs = max(len(values) for values in loss_logs.values())
    tick_spacing = max(1, max_epochs // 10)  # Show at most 10 major ticks
    epochs = list(range(0, max_epochs + 1, tick_spacing))
    plt.xticks(epochs, [str(e) for e in epochs])

    # Round tick interval to nearest 0.05, 0.1, 0.5, or 1
    def round_tick_interval(value):
        if value >= 1:
            return round(value)  # Round to whole numbers
        elif value >= 0.5:
            return round(value * 2) / 2  # Round to nearest 0.5
        elif value >= 0.1:
            return round(value * 10) / 10  # Round to nearest 0.1
        else:
            return round(value * 20) / 20  # Round to nearest 0.05

    y_range = ymax - ymin
    base_tick_interval = y_range / 10  # Approximate tick interval for ~10 ticks
    tick_interval = round_tick_interval(base_tick_interval)

    # Generate y-ticks that end with 0 or 5
    y_ticks = np.arange(np.ceil(ymin / tick_interval) * tick_interval,
                        np.floor(ymax / tick_interval) * tick_interval + tick_interval,
                        tick_interval)

    # Ensure ticks end in 0 or 5 by rounding
    y_ticks = np.array([round(y_tick * 20) / 20 for y_tick in y_ticks])
    # Ensure all y-tick labels have the same decimal places
    decimal_places = 0
    if tick_interval < 1:
        decimal_places = abs(int(np.log10(tick_interval)))  # Count decimal places
    y_tick_labels = [f"{tick:.{decimal_places}f}" for tick in y_ticks]
    plt.yticks(y_ticks, y_tick_labels)

    plt.legend()
    plt.grid(True, linestyle='dashed', linewidth=0.5, alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_hits_curve(hits_logs: dict,
                    save_path: Optional[Union[str, Path]] = None,
                    title: str = 'Hits@K Curve',
                    xlabel: str = 'Epochs',
                    ylabel: str = 'Hits@K'):
    """
    Plot the Hits@K curve along training (optimization).

    Parameters
    ----------
    hits_logs : list[dict]
        List of Hits@K scores for each epoch.
    save_path : str or Path, optional
        Path to save the plot. If None, the plot will not be saved. Default is None.
    title : str, optional
        Title of the plot. Default is 'Hits@K Curve'.
    xlabel : str, optional
        Label for the x-axis. Default is 'Epochs'.
    ylabel : str, optional
        Label for the y-axis. Default is 'Hits@K'.
    Returns
    -------
    None
    """

    plt.figure(figsize=(8, 6))

    for key, values in hits_logs.items():
        plt.plot(range(1, len(values) + 1), values, label=f"Hits@{key}", linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(-0.05, 1.05)  # Set y-limits to be between -0.05 and 1.05

    max_epochs = max(len(values) for values in hits_logs.values())
    tick_spacing = max(1, max_epochs // 10)  # Show at most 10 major ticks
    epochs = list(range(0, max_epochs + 1, tick_spacing))
    plt.xticks(epochs, [str(e) for e in epochs])
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{i/10:.1f}" for i in range(11)])  # Set y-ticks from 0 to 1 with step of 0.1

    plt.legend()
    plt.grid(True, linestyle='dashed', linewidth=0.5, alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_mrr_curve(mrr_logs: Union[list, np.ndarray],
                   save_path: Optional[Union[str, Path]] = None,
                   title: str = 'MRR Curve',
                   xlabel: str = 'Epochs',
                   ylabel: str = 'MRR',
                   xlim=None,
                   ylim=None,
                   grid: bool = True):
    """
    Plot the MRR curve along training (optimization).

    Parameters
    ----------
    mrr_logs : list or np.ndarray
       List of MRR values for each epoch.
    save_path : str or Path, optional
       Path to save the plot. If None, the plot will not be saved. Default is None.
    title : str, optional
       Title of the plot. Default is 'MRR Curve'.
    xlabel : str, optional
       Label for the x-axis. Default is 'Epochs'.
    ylabel : str, optional
       Label for the y-axis. Default is 'MRR'.
    xlim : tuple[float, float], optional
       Limits for the x-axis. Default is None.
    ylim : tuple[float, float], optional
       Limits for the y-axis. Default is None.
    grid : bool, optional
       Whether to show grid lines. Default is True.

    Returns
    -------
    None
    """

    plt.plot(range(1, len(mrr_logs) + 1), mrr_logs, linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(-0.05, 1.05)  # Set y-limits to be between -0.05 and 1.05

    max_epochs = len(mrr_logs)
    tick_spacing = max(1, max_epochs // 10)  # Show at most 10 major ticks
    epochs = list(range(0, max_epochs + 1, tick_spacing))
    plt.xticks(epochs, [str(e) for e in epochs])
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{i / 10:.1f}" for i in range(11)])  # Set y-ticks from 0 to 1 with step of 0.1

    plt.grid(True, linestyle='dashed', linewidth=0.5, alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Example usage
    example_loss_logs = {
        '1': np.sort(np.random.rand(200)),
        '10': np.sort(np.random.rand(200)),
    }
    example_mrr_logs = np.sort(np.random.rand(200))

    # plot_hits_curve(example_loss_logs)
    plot_mrr_curve(example_mrr_logs)
