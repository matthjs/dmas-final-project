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
