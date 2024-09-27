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
                   - network: Optional[nx.Graph] (if a predefined network is provided)
                   - steps: int (only required for simulation mode)
                   - mode: str ("interactive" or "simulation")

    :return: An instance of the NewsMediaModel initialized with the provided parameters.
    """
    num_users = params.get('num_users')
    num_official_media = params.get('num_official_media')
    num_self_media = params.get('num_self_media')
    opinion_dims = params.get('opinion_dims')
    network_type = params.get('network_type')
    network_params = params.get('network_params')
    align_freq = params.get('align_freq', 10)
    network = params.get('network')  # Predefined network if provided

    # Instantiate the model
    model = NewsMediaModel(
        num_users=num_users,
        num_official_media=num_official_media,
        num_self_media=num_self_media,
        opinion_dims=opinion_dims,
        network_type=network_type,
        network_params=network_params,
        align_freq=align_freq
    )

    # If a predefined network is provided, override the generated network
    if network is not None:
        pass
        # TODO:

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
