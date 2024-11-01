import threading
from typing import Union, SupportsFloat, Any, Dict, Tuple
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from typing import Tuple

"""
Online algorithms for keeping track of sample means and sample variances.

For more information, see: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

.. note::
   This module implements online algorithms for calculating mean, variance, and sample variance.
"""


def default_welford_dict():
    """Returns a defaultdict for storing Welford objects."""
    return defaultdict(dict)


class Welford:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_aggr(self, new_val: float) -> None:
        self.count += 1
        delta = new_val - self.mean
        self.mean += delta / self.count
        delta2 = new_val - self.mean
        self.M2 += delta * delta2

    def get_curr_mean_variance(self) -> Tuple[float, float]:
        mean, _, var = self._finalize_aggr()
        return mean, var

    def _finalize_aggr(self) -> Tuple[float, float, float]:
        if self.count < 2:
            return self.mean, 0, 0
        else:
            variance = self.M2 / self.count
            sample_variance = self.M2 / (self.count - 1)
            return self.mean, variance, sample_variance


class MetricsTracker:
    """
    Object for recording various metrics across multiple runs.
    Tracks the mean and standard deviation of each metric using Welford's algorithm,
    averaged over multiple runs.
    """

    def __init__(self):
        # Replace lambda with helper function to avoid pickling issues
        self._metrics_history: Dict[str, Dict[str, Dict[int, Welford]]] = defaultdict(default_welford_dict)
        self.register_metric("loss")
        self.register_metric("return")

    def register_metric(self, metric_name: str) -> None:
        """
        Register a new metric to be tracked.

        :param metric_name: The name of the metric to register (e.g., "return", "accuracy").
        """
        if metric_name not in self._metrics_history:
            self._metrics_history[metric_name] = defaultdict(dict)

    def record_metric(self, metric_name: str, agent_id: str, episode_idx: int,
                      value: Union[float, int, SupportsFloat]) -> None:
        """
        Record a value for a specific metric, agent, and episode. Uses Welford's algorithm to update mean and variance.

        :param metric_name: The name of the metric (e.g., "return", "accuracy").
        :param agent_id: The identifier of the agent.
        :param episode_idx: The index of the episode.
        :param value: The metric value to record.
        """
        if episode_idx not in self._metrics_history[metric_name][agent_id]:
            self._metrics_history[metric_name][agent_id][episode_idx] = Welford()

        self._metrics_history[metric_name][agent_id][episode_idx].update_aggr(float(value))

    def get_mean_std(self, metric_name: str, agent_id: str, episode_idx: int) -> Tuple[Any, Any]:
        """
        Get the latest recorded mean and standard deviation for a specific metric, agent, and episode.

        :param metric_name: The name of the metric.
        :param agent_id: The identifier of the agent.
        :param episode_idx: The episode index.
        :return: The mean and standard deviation for the metric, or (None, None) if no values have been recorded.
        """
        if agent_id in self._metrics_history[metric_name] and episode_idx in self._metrics_history[metric_name][
            agent_id]:
            welford = self._metrics_history[metric_name][agent_id][episode_idx]
            mean, var = welford.get_curr_mean_variance()
            return mean, np.sqrt(var)
        else:
            return None, None

    def plot_metric(self, metric_name: str, file_name: str, num_episodes: int = None, x_axis_label="Time Steps",
                    y_axis_label="Metric Value", title=None) -> None:
        """
        Plot the average value and standard deviation over episodes for a specific metric across multiple runs.

        :param metric_name: The name of the metric to plot (e.g., "return", "accuracy").
        :param file_name: The file to save the plot.
        :param num_episodes: The number of episodes to plot.
        :param x_axis_label: The label for the x-axis.
        :param y_axis_label: The label for the y-axis.
        :param title: The title of the plot (optional).
        """
        if metric_name not in self._metrics_history:
            raise ValueError(f"Metric '{metric_name}' not found")

        fig, ax = plt.subplots(figsize=(10, 8))

        for agent_id, episode_welfords in self._metrics_history[metric_name].items():
            sorted_episodes = sorted(episode_welfords.items())

            if num_episodes:
                sorted_episodes = sorted_episodes[:num_episodes]

            mean_values = [w.get_curr_mean_variance()[0] for _, w in sorted_episodes]
            std_values = [np.sqrt(w.get_curr_mean_variance()[1]) for _, w in sorted_episodes]

            window = 20
            mean_values = pd.Series(mean_values).rolling(window=window, min_periods=1).mean()
            std_values = pd.Series(std_values).rolling(window=window, min_periods=1).mean()

            x_values = np.array([ep for ep, _ in sorted_episodes])

            ax.plot(x_values, mean_values, label=f'{agent_id} agent')
            ax.fill_between(x_values,
                            np.array(mean_values) - 0.1 * np.array(std_values),
                            np.array(mean_values) + 0.1 * np.array(std_values),
                            alpha=0.2)

        ax.set_title(title if title else f'{metric_name.capitalize()} History', fontsize=16)
        ax.set_xlabel(x_axis_label, fontsize=14)
        ax.set_ylabel(y_axis_label, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(file_name)

    def clear_metrics(self) -> None:
        """
        Clear the recorded metrics for all agents and all metrics.
        """
        self._metrics_history.clear()

    @property
    def metrics_history(self) -> dict:
        """
        Get the entire history of recorded metrics.

        :return: A dictionary containing the recorded metric values for each agent and episode.
        """
        return dict(self._metrics_history)
