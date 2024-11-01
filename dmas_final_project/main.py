import pickle
import numpy as np
import random
from dmas_final_project.data_processing.data_aggregator import RunDataAggregator
from dmas_final_project.data_processing.metrics_tracker import MetricsTracker
from dmas_final_project.parser.parser import parse_arguments, parse_news_media_model
from dmas_final_project.plotting.plotting import plot_evolution_by_dimension, \
    plot_individual_alignment_over_time, plot_metrics, \
    plot_opinion_or_bias_frequency
from dmas_final_project.view.view import get_server
import os


def save_data(metric_tracker: MetricsTracker, aggr: RunDataAggregator, path: str) -> None:
    """
    Save the MetricsTracker and RunDataAggregator objects directly to files.

    :param metric_tracker: The tracker recording metrics over runs.
    :param aggr: The data aggregator for collected run data.
    :param path: The directory path to save the data files.
    """
    # Save metric tracker
    with open(os.path.join(path, 'metric_tracker.pkl'), 'wb') as f:
        pickle.dump(metric_tracker, f)

    # Save aggregated run data
    with open(os.path.join(path, 'aggr.pkl'), 'wb') as f:
        pickle.dump(aggr, f)


def main() -> None:
    """
    Main function to parse arguments and run the agent-based model.
    """
    params, args = parse_arguments()

    # Ensure the save path exists
    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)

    model = None

    if params['mode'] == 'interactive':
        server = get_server(params)
        server.launch()
    elif params['mode'] == 'simulation':
        print("Running simulation...")

        # Fix seed to get reproducible results.
        random.seed(42)
        np.random.seed(42)
        num_runs = 10        # This is a hardcoded value.
        metric_tracker = MetricsTracker()
        aggr = RunDataAggregator()
        for run in range(num_runs):
            model = parse_news_media_model(params)
            model.set_metrics_tracker(metric_tracker)
            for _ in range(params['steps']):
                model.step()
            print("Simulation complete.")

            aggr.add_run_data(model.datacollector)
            if run == num_runs - 1:
                plot_evolution_by_dimension(model.datacollector, data_column='Opinion', label='Opinion')
                if params.get('num_self_media') > 0:
                    plot_evolution_by_dimension(model.datacollector, data_column='Bias', label='Bias')
                plot_individual_alignment_over_time(model.datacollector)

        plot_opinion_or_bias_frequency(aggr, 'opinion', path=args.plot_path)
        if params.get('num_self_media') > 0:
            plot_opinion_or_bias_frequency(aggr, 'bias', path=args.plot_path)
        plot_metrics(metric_tracker, path=args.plot_path)

        if model is not None:
            # Save metric tracker and aggregated data
            save_data(metric_tracker, aggr, args.plot_path)


if __name__ == "__main__":
    """
    python main.py interactive --config_file <config.json> --network_file <network_file>

    python main.py simulation --config_file config.json --steps 2000 --plot_path <path>
    """
    main()
