import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from mesa_viz_tornado.ModularVisualization import ModularServer
from mesa_viz_tornado.modules import NetworkModule, ChartModule

from dmas_final_project.agents.official_news_agent import OfficialNewsAgent
from dmas_final_project.agents.self_news_agent import SelfNewsAgent
from dmas_final_project.agents.user_agent import UserAgent
from dmas_final_project.models.news_media_model import NewsMediaModel
from dmas_final_project.parser.parser import parse_arguments, parse_news_media_model
from dmas_final_project.plotting.plotting import plot_global_alignment_over_time
from dmas_final_project.view.view import network_portrayal


def plot_opinion_dynamics(results):
    """
    Plot the opinion dynamics for each agent over time.

    :param results: DataFrame containing agent data collected over time.
    """
    # Plot opinion dynamics for each agent over time
    plt.figure(figsize=(10, 6))

    # Loop through each agent and plot their opinion dynamics
    for agent_id in results.index.get_level_values(1).unique():
        agent_data = results.xs(agent_id, level=1)

        # Ensure 'Opinion' column is free of None values before applying np.mean
        filtered_opinions = agent_data['Opinion'].dropna().apply(lambda x: np.mean(x) if x is not None else np.nan)

        plt.plot(filtered_opinions.index, filtered_opinions, label=f"Agent {agent_id}")

    plt.title('Opinion Dynamics Over Time')
    plt.xlabel('Step')
    plt.ylabel('Average Opinion')
    plt.legend()
    plt.savefig("opinion_dynamics.png")
    plt.show()


def visualize_social_network(model):
    """
    Visualize the social network graph where nodes represent agents
    and edges represent connections between them.

    :param model: The NewsMediaModel instance containing the network.
    """
    plt.figure(figsize=(12, 12))

    # Get the network graph from the model
    G = model.G

    # Assign colors to different types of agents
    color_map = []
    for node in G:
        agent = G.nodes[node]['agent']
        if isinstance(agent, UserAgent):
            color_map.append('blue')
        elif isinstance(agent, OfficialNewsAgent):
            color_map.append('green')
        elif isinstance(agent, SelfNewsAgent):
            color_map.append('red')

    # Draw the network
    pos = nx.spring_layout(G)  # Spring layout for better visualization
    nx.draw(G, pos, node_color=color_map, with_labels=False, node_size=100, alpha=0.8, edge_color='gray')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='User Agent', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Official News Agent', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Self News Agent', markerfacecolor='red', markersize=10)]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title('Social Network Visualization')
    plt.savefig("graph.png")
    plt.show()


def run() -> None:
    # Hardcoded model parameters for now.
    num_users = 50  # Number of user agents
    num_official_media = 1  # Number of official news agents
    num_self_media = 5  # Number of self-news agents
    opinion_dims = 3  # Number of opinion dimensions (multi-dimensional opinion space)

    # Define network parameters (scale-free network)
    network_type = 'scale_free'  # Choose between 'scale_free' or 'small_world'
    network_params = {'n': num_users + num_official_media + num_self_media,
                      'm': 2}  # Parameters for Barabási-Albert model

    # Initialize the model
    model = NewsMediaModel(
        num_users=num_users,
        num_official_media=num_official_media,
        num_self_media=num_self_media,
        opinion_dims=opinion_dims,
        network_type=network_type,
        network_params=network_params,
        align_freq=10
    )

    # Run the model for a specified number of steps
    # num_steps = 1000
    # for i in range(num_steps):
    #     model.step()

    # Collect the results
    # results = model.datacollector.get_agent_vars_dataframe()
    # print(results)

    # plot_opinion_dynamics(results)
    # visualize_social_network(model)
    # plot_global_alignment_over_time(model)


def main() -> None:
    """
    Main function to parse arguments and run the agent-based model.
    """
    params = parse_arguments()

    if params['mode'] == 'interactive':
        network = NetworkModule(network_portrayal, 500, 500)
        chart = ChartModule([{"Label": "Global Alignment", "Color": "Blue"}])

        server = ModularServer(NewsMediaModel,
                               [network, chart],
                               "News Media Model",
                               {"num_users": 50,
                                "num_official_media": 1,
                                "num_self_media": 5,
                                "opinion_dims": 3,
                                "network_type": 'scale_free',
                                "network_params": {"n": 56, "m": 2},
                                "align_freq": 10})
        server.port = 8521
        server.launch()
    elif params['mode'] == 'simulation':
        print("Running simulation...")
        model = parse_news_media_model(params)
        for _ in range(params['steps']):
            model.step()
        print("Simulation complete.")
        plot_global_alignment_over_time(model)


if __name__ == "__main__":
    """
    python run_model.py interactive --config_file <config.json> --network_file <network_file>
    
    python run_model.py simulation --config_file config.json --steps 200
    """
    main()
