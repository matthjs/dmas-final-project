from mesa_viz_tornado.ModularVisualization import ModularServer
from mesa_viz_tornado.UserParam import Slider
from mesa_viz_tornado.modules import NetworkModule, ChartModule

from dmas_final_project.agents.self_news_agent import SelfNewsAgent
from dmas_final_project.agents.user_agent import UserAgent
from typing import Dict, Any
import networkx as nx
from mesa import Agent

from dmas_final_project.models.news_media_model import NewsMediaModel


def agent_portrayal(agent: Agent) -> Dict[str, Any]:
    """
    Generate the visual portrayal dictionary for a given agent.

    :param agent: The agent to generate the portrayal for.

    :return: A dictionary representing the visual attributes of the agent.
    """
    portrayal = {
        "Shape": "circle",
        "Color": "blue" if isinstance(agent, UserAgent) else "red" if isinstance(agent, SelfNewsAgent) else "green",
        "r": 0.5,
        "Layer": 0,
        "Filled": "true",
        "text": f"ID: {agent.unique_id}",
        "text_color": "white",
        "stroke_color": "black",  # Adds a border around the node for better visibility
        "stroke_width": 1.5,  # Adjusts the thickness of the border
    }
    return portrayal


def network_portrayal(G: nx.Graph) -> Dict[str, Any]:
    """
    Generate the visual portrayal dictionary for a network graph.

    :param G: The graph representing the network of agents.

    :return: A dictionary containing lists of node and edge portrayals.
    """
    portrayal = {
        'nodes': [],
        'edges': []
    }

    for node_id, data in G.nodes(data=True):
        agents = data.get('agent', [])
        if not isinstance(agents, list):
            agents = [agents]  # Ensure agents is a list even if it contains a single agent

        for agent in agents:
            node_portrayal = agent_portrayal(agent)
            node_portrayal.update({
                "id": node_id,
                "size": 7,  # Example size; you might want to adjust based on agent properties
                "color": node_portrayal["Color"],  # Color is already set in agent_portrayal
                "label": node_portrayal["text"]
            })
            portrayal['nodes'].append(node_portrayal)

    for source, target in G.edges:
        portrayal['edges'].append({
            'source': source,
            'target': target,
            'width': 2,  # Set the width of the edges
            'color': "gray",  # You can also set different colors for edges if needed
        })

    return portrayal


def get_server(params: Dict[str, Any]) -> ModularServer:
    network = NetworkModule(network_portrayal, 500, 500)
    chart = ChartModule([{"Label": "Global Alignment", "Color": "Blue"}])
    server = ModularServer(
        model_cls=NewsMediaModel,
        visualization_elements=[network, chart],
        name="News Media Model",
        model_params={
            "num_users": Slider("Number of Users", value=params["num_users"], min_value=10, max_value=200, step=1),
            "num_official_media": Slider("Number of Official Media", value=params["num_official_media"], min_value=1, max_value=10, step=1),
            "num_self_media": Slider("Number of Self-Media", value=params["num_self_media"], min_value=0, max_value=20, step=1),
            "opinion_dims": Slider("Opinion Dimensions", value=params["opinion_dims"], min_value=1, max_value=10, step=1),
            "network_type": params["network_type"],
            "network_params_m": Slider("Number of Edges per New Node (m)", value=params["network_params"]["m"], min_value=1, max_value=10, step=1),
            "align_freq": Slider("Alignment Frequency", value=params["align_freq"], min_value=1, max_value=100, step=1),
            "opinion_mean": Slider("Opinion Mean", value=params["opinion_mean"], min_value=-1, max_value=1, step=0.1),
            "opinion_std": Slider("Opinion StdDev", value=params["opinion_std"], min_value=0, max_value=2, step=0.1),
            "user_rationality_mean": Slider("User Rationality Mean", value=params["user_rationality_mean"], min_value=0, max_value=1, step=0.1),
            "user_rationality_std": Slider("User Rationality StdDev", value=params["user_rationality_std"], min_value=0, max_value=0.5, step=0.05),
            "user_affective_involvement_mean": Slider("User Affective Involvement Mean", value=params["user_affective_involvement_mean"], min_value=0, max_value=1, step=0.1),
            "user_affective_involvement_std": Slider("User Affective Involvement StdDev", value=params["user_affective_involvement_std"], min_value=0, max_value=0.5, step=0.05),
            "user_tolerance_threshold_mean": Slider("User Tolerance Threshold Mean", value=params["user_tolerance_threshold_mean"], min_value=0, max_value=1, step=0.1),
            "user_tolerance_threshold_std": Slider("User Tolerance Threshold StdDev", value=params["user_tolerance_threshold_std"], min_value=0, max_value=0.5, step=0.05),
            "self_media_bias_mean": Slider("Self Media Bias Mean", value=params["self_media_bias_mean"], min_value=-1, max_value=1, step=0.1),
            "self_media_bias_std": Slider("Self Media Bias StdDev", value=params["self_media_bias_std"], min_value=0, max_value=1, step=0.1),
            "self_media_adjustability_mean": Slider("Self Media Adjustability Mean", value=params["self_media_adjustability_mean"], min_value=0, max_value=1, step=0.1),
            "self_media_adjustability_std": Slider("Self Media Adjustability StdDev", value=params["self_media_adjustability_std"], min_value=0, max_value=0.5, step=0.05),
            "network_params": params["network_params"]
        }
    )

    server.port = 8521

    return server
