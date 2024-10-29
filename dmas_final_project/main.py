import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from mesa_viz_tornado.ModularVisualization import ModularServer
from mesa_viz_tornado.modules import NetworkModule, ChartModule
import random
from dmas_final_project.agents.official_news_agent import OfficialNewsAgent
from dmas_final_project.agents.self_news_agent import SelfNewsAgent
from dmas_final_project.agents.user_agent import UserAgent
from dmas_final_project.data_processing.data_aggregator import RunDataAggregator
from dmas_final_project.data_processing.metrics_tracker import MetricsTracker
from dmas_final_project.models.news_media_model import NewsMediaModel
from dmas_final_project.parser.parser import parse_arguments, parse_news_media_model
from dmas_final_project.plotting.plotting import plot_global_alignment_over_time, plot_evolution_by_dimension, \
    plot_individual_alignment_over_time, plot_polarization, plot_social_network, plot_metrics, plot_opinion_frequency
from dmas_final_project.view.view import network_portrayal, get_server


def main() -> None:
    """
    Main function to parse arguments and run the agent-based model.
    """
    params = parse_arguments()

    if params['mode'] == 'interactive':
        server = get_server(params)
        server.launch()
    elif params['mode'] == 'simulation':
        print("Running simulation...")

        random.seed(5)
        np.random.seed(5)
        num_runs = 10
        metric_tracker = MetricsTracker()
        aggr = RunDataAggregator()
        for run in range(num_runs):
            # Make sure to fix the seed so the social network is the same for each run!
            model = parse_news_media_model(params)
            model.set_metrics_tracker(metric_tracker)
            for _ in range(params['steps']):
                model.step()
            print("Simulation complete.")

            aggr.add_run_data(model.datacollector)
            if run == num_runs - 1:
                plot_social_network(model)
                plot_global_alignment_over_time(model.datacollector)
                plot_evolution_by_dimension(model.datacollector, data_column='Opinion', label='Opinion')
                plot_evolution_by_dimension(model.datacollector, data_column='Bias', label='Bias')
                plot_individual_alignment_over_time(model.datacollector)
                plot_polarization(model.datacollector)

        plot_opinion_frequency(aggr)
        plot_metrics(metric_tracker)



if __name__ == "__main__":
    """
    python main.py interactive --config_file <config.json> --network_file <network_file>
    
    python main.py simulation --config_file config.json --steps 1000
    """
    main()
