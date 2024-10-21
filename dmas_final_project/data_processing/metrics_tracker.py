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


class Welford:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_aggr(self, new_val: float) -> None:
        """
        Update aggregate values with a new value.

        :param new_val: The new value to be incorporated into the aggregate.
        """
        self.count += 1
        delta = new_val - self.mean
        self.mean += delta / self.count
        delta2 = new_val - self.mean
        self.M2 += delta * delta2

    def get_curr_mean_variance(self) -> Tuple[float, float]:
        """
        Return the current sample mean and sample variance estimate.
        :return: mean and variance.
        """
        mean, _, var = self._finalize_aggr()
        return mean, var

    def _finalize_aggr(self) -> Tuple[float, float, float]:
        """
        Retrieve the mean, variance, and sample variance from the aggregate.

        :return: Mean, variance, and sample variance calculated from the aggregate.
        """
        if self.count < 2:
            return self.mean, 0, 0
        else:
            variance = self.M2 / self.count
            sample_variance = self.M2 / (self.count - 1)
            return self.mean, variance, sample_variance


class MetricsTracker:
    """
    Thread-safe object for recording various metrics across multiple runs.
    Tracks the mean and standard deviation of each metric using Welford's algorithm,
    averaged over multiple runs.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Dictionary of metric histories where key is metric_name, value is agent_id -> episode_idx -> Welford object
        self._metrics_history: Dict[str, Dict[str, Dict[int, Welford]]] = defaultdict(lambda: defaultdict(dict))

        self.register_metric("loss")
        self.register_metric("return")

    def register_metric(self, metric_name: str) -> None:
        """
        Register a new metric to be tracked.

        :param metric_name: The name of the metric to register (e.g., "return", "accuracy").
        """
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            if agent_id in self._metrics_history[metric_name] and episode_idx in self._metrics_history[metric_name][
                agent_id]:
                welford = self._metrics_history[metric_name][agent_id][episode_idx]
                mean, var = welford.get_curr_mean_variance()
                return mean, np.sqrt(var)
            else:
                return None, None

    def plot_metric(self, metric_name: str, file_name: str, num_episodes: int = None, x_axis_label="Episodes",
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
        with self._lock:
            if metric_name not in self._metrics_history:
                raise ValueError(f"Metric '{metric_name}' not found")

            fig, ax = plt.subplots(figsize=(10, 8))

            for agent_id, episode_welfords in self._metrics_history[metric_name].items():
                # Sort by episode index to ensure proper plotting order
                sorted_episodes = sorted(episode_welfords.items())

                if num_episodes:
                    sorted_episodes = sorted_episodes[:num_episodes]

                mean_values = [w.get_curr_mean_variance()[0] for _, w in sorted_episodes]
                std_values = [np.sqrt(w.get_curr_mean_variance()[1]) for _, w in sorted_episodes]

                # Apply a smoothing on the data with a running window average over episodes
                window = 20
                mean_values = pd.Series(mean_values).rolling(window=window, min_periods=1).mean()
                std_values = pd.Series(std_values).rolling(window=window, min_periods=1).mean()

                x_values = np.array([ep for ep, _ in sorted_episodes])  # X-axis corresponds to episode indices

                # Plot the mean value for each episode
                ax.plot(x_values, mean_values, label=f'{agent_id} agent')

                # Fill between mean ± standard deviation
                ax.fill_between(x_values,
                                np.array(mean_values) - np.array(std_values),
                                np.array(mean_values) + np.array(std_values),
                                alpha=0.2)

            ax.set_title(title if title else f'{metric_name.capitalize()} History')
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            plt.savefig(file_name)

    def clear_metrics(self) -> None:
        """
        Clear the recorded metrics for all agents and all metrics.
        """
        with self._lock:
            self._metrics_history.clear()

    @property
    def metrics_history(self) -> dict:
        """
        Get the entire history of recorded metrics.

        :return: A dictionary containing the recorded metric values for each agent and episode.
        """
        with self._lock:
            return dict(self._metrics_history)