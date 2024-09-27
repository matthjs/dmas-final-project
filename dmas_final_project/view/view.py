from dmas_final_project.agents.self_news_agent import SelfNewsAgent
from dmas_final_project.agents.user_agent import UserAgent
from typing import Dict, Any
import networkx as nx
from mesa import Agent


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
    portrayal = dict()
    portrayal['nodes'] = [agent_portrayal(node) for node in G.nodes]
    portrayal['edges'] = [{'source': source.unique_id, 'target': target.unique_id} for source, target in G.edges]
    return portrayal
