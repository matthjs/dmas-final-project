import argparse
import json
from typing import Dict, Any
import networkx as nx
from mesa.space import NetworkGrid

from dmas_final_project.models.news_media_model import NewsMediaModel


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command-line arguments for running the NewsMediaModel.

    :return: A dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run the NewsMediaModel either interactively or as a simulation.")

    parser.add_argument('mode', choices=['interactive', 'simulation'],
                        help="Choose the mode to run: 'interactive' or 'simulation'.")

    parser.add_argument('--config_file', type=str, required=True,
                        help="Path to the JSON file containing model configuration parameters.")

    parser.add_argument('--network_file', type=str, required=False,
                        help="Path to the file containing the network structure (edgelist or adjacency list).")

    parser.add_argument('--steps', type=int, default=100,
                        help="Number of steps to run in 'simulation' mode. Default is 100.")

    args = parser.parse_args()

    # Load configuration parameters from JSON file
    with open(args.config_file, 'r') as file:
        config_params = json.load(file)

    # Load the network structure if provided
    if args.network_file:
        network = load_network(args.network_file)
    else:
        network = None

    config_params['mode'] = args.mode
    config_params['steps'] = args.steps
    config_params['network'] = network

    return config_params


def parse_news_media_model(params: Dict[str, Any]) -> NewsMediaModel:
    """
    Instantiate the NewsMediaModel based on the provided parameters.

    :param params: Dictionary containing the parameters for the model.
                   Required keys are:
                   - num_users: int
                   - num_official_media: int
                   - num_self_media: int
                   - opinion_dims: int
                   - network_type: str
                   - network_params: dict
                   - align_freq: int
                   - extra_media_edges: int
                   - extra_self_media_edges: int
                   - enable_feedback: bool
                   - steps: int (only required for simulation mode)
                   - mode: str ("interactive" or "simulation")

    :return: An instance of the NewsMediaModel initialized with the provided parameters.
    """

    # Fetch the mandatory parameters from the params dictionary
    num_users = params.get('num_users')
    num_official_media = params.get('num_official_media')
    num_self_media = params.get('num_self_media')
    opinion_dims = params.get('opinion_dims')
    network_type = params.get('network_type')
    network_params = params.get('network_params')
    align_freq = params.get('align_freq', 10)  # Default value of 10 if not provided
    extra_media_edges = params.get('extra_media_edges', 0)  # Default to 0
    extra_self_media_edges = params.get('extra_self_media_edges', 0)  # Default to 0
    enable_feedback = params.get('enable_feedback', True)

    # Opinion distribution parameters
    opinion_mean = params.get('opinion_mean', 0)
    opinion_std = params.get('opinion_std', 1)

    # User behavior parameters
    user_rationality_mean = params.get('user_rationality_mean', 0.5)
    user_rationality_std = params.get('user_rationality_std', 0.1)
    user_affective_involvement_mean = params.get('user_affective_involvement_mean', 0.5)
    user_affective_involvement_std = params.get('user_affective_involvement_std', 0.1)
    user_tolerance_threshold_mean = params.get('user_tolerance_threshold_mean', 0.5)
    user_tolerance_threshold_std = params.get('user_tolerance_threshold_std', 0.1)

    # Self-media parameters
    self_media_bias_mean = params.get('self_media_bias_mean', 0)
    self_media_bias_std = params.get('self_media_bias_std', 0.5)
    self_media_adjustability_mean = params.get('self_media_adjustability_mean', 0.1)
    self_media_adjustability_std = params.get('self_media_adjustability_std', 0.05)

    # Optional predefined network (if provided)
    network = params.get('network')

    # Instantiate the NewsMediaModel
    model = NewsMediaModel(
        num_users=num_users,
        num_official_media=num_official_media,
        num_self_media=num_self_media,
        opinion_dims=opinion_dims,
        network_type=network_type,
        network_params=network_params,
        align_freq=align_freq,
        extra_media_edges=extra_media_edges,
        extra_self_media_edges=extra_self_media_edges,
        enable_feedback=enable_feedback,
        opinion_mean=opinion_mean,
        opinion_std=opinion_std,
        user_rationality_mean=user_rationality_mean,
        user_rationality_std=user_rationality_std,
        user_affective_involvement_mean=user_affective_involvement_mean,
        user_affective_involvement_std=user_affective_involvement_std,
        user_tolerance_threshold_mean=user_tolerance_threshold_mean,
        user_tolerance_threshold_std=user_tolerance_threshold_std,
        self_media_bias_mean=self_media_bias_mean,
        self_media_bias_std=self_media_bias_std,
        self_media_adjustability_mean=self_media_adjustability_mean,
        self_media_adjustability_std=self_media_adjustability_std
    )

    # If a predefined network is provided, override the model's generated network
    if network is not None:
        model.G = network
        model.grid = NetworkGrid(model.G)

    return model


def load_network(network_file: str) -> nx.Graph:
    """
    Load a network structure from a file.

    :param network_file: Path to the network file (edgelist or adjacency list).
    :return: A NetworkX graph object.
    """
    # Assuming the network file is an edgelist; you can modify this to support other formats.
    G = nx.read_edgelist(network_file, nodetype=int)
    return G
