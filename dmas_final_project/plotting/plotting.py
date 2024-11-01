from collections import defaultdict
from dmas_final_project.data_processing.data_aggregator import RunDataAggregator
import matplotlib.pyplot as plt
import networkx as nx
from mesa import DataCollector
from dmas_final_project.agents.official_news_agent import OfficialNewsAgent
from dmas_final_project.agents.self_news_agent import SelfNewsAgent
from dmas_final_project.agents.user_agent import UserAgent
from dmas_final_project.data_processing.metrics_tracker import MetricsTracker
from dmas_final_project.models.news_media_model import NewsMediaModel
import numpy as np
import os


def plot_metrics(metric_tracker: MetricsTracker, path) -> None:
    """
    Plot the global alignment and polarization across multiple runs.

    :param model: The NewsMediaModel instance with metrics tracked.
    """
    # Plot global alignment
    metric_tracker.plot_metric(
        metric_name='Global Alignment',
        file_name=os.path.join(path, 'global_alignment_across_runs.svg'),
        title='Global Alignment Across Runs'
    )

    # Plot polarization
    metric_tracker.plot_metric(
        metric_name='Polarization',
        file_name=os.path.join(path, 'polarization_across_runs.svg'),
        title='Polarization Across Runs'
    )

    metric_tracker.plot_metric(
        metric_name='Homophily Index',
        file_name=os.path.join(path, 'Homophily_Index+acrosss_runs.svg'),
        title='Homophily Index Across Runs'
    )

    metric_tracker.plot_metric(
        metric_name='Mean Opinion Magnitude',
        file_name=os.path.join(path, 'mean_accross_runs.svg'),
        title='Mean Opinion Magnitude Across Runs'
    )



def plot_global_alignment_over_time(datacollector: DataCollector) -> None:
    """
    Plot the global alignment over time using data collected by the DataCollector.

    :param datacollector: The DataCollector instance containing the global alignment data.
    """
    # Retrieve the global alignment data from the DataCollector
    global_alignment_data = datacollector.get_model_vars_dataframe()

    if global_alignment_data.empty:
        print("No alignment data available. Ensure the model has been stepped through multiple iterations.")
        return

    # Plotting the global alignment over time
    plt.figure(figsize=(10, 6))
    plt.plot(global_alignment_data.index, global_alignment_data['Global Alignment'], marker='o', linestyle='-',
             color='b')
    plt.xlabel('Time Step')
    plt.ylabel('Global Alignment (A(t))')
    plt.title('Global Alignment Over Time')
    plt.grid(True)
    plt.savefig("alignments.png")
    plt.show()


def plot_individual_alignment_over_time(datacollector: DataCollector) -> None:
    """
    Plot the alignment evolution over time for each agent using data collected by the DataCollector.

    :param datacollector: The DataCollector instance containing the agent alignment data.
    """
    # Retrieve the agent data from the DataCollector
    agent_data = datacollector.get_agent_vars_dataframe()

    if agent_data.empty:
        print("No alignment data available. Ensure the model has been stepped through multiple iterations.")
        return

    # Ensure 'Alignment' is a column in the agent data and extract that column
    try:
        alignment_data = agent_data['Alignment']
    except KeyError:
        print("'Alignment' column not found in the agent data.")
        return

    # Remove rows with None values in the 'Alignment' column (since alignment is computed less frequently)
    alignment_data = alignment_data.dropna()

    # Plotting the alignment over time for each agent
    plt.figure(figsize=(10, 6))

    # Iterate over unique agents
    for agent_id in alignment_data.index.get_level_values('AgentID').unique():
        # Extract the alignment data for the current agent
        agent_alignment_data = alignment_data.xs(agent_id, level='AgentID')

        # Plot the alignment over time for this agent
        plt.plot(agent_alignment_data.index, agent_alignment_data,
                 label=f'Agent {agent_id}')

    plt.xlabel('Time Step')
    plt.ylabel('Alignment')
    plt.title('Individual Alignment Over Time')
    plt.grid(True)
    # plt.legend()
    plt.savefig("individual_alignments.png")
    plt.show()


def plot_evolution_by_dimension(datacollector: DataCollector, data_column: str, label: str) -> None:
    """
    Plot the evolution of values over time for each agent on each dimension (Opinion or Bias).

    :param datacollector: The DataCollector instance containing the agent data.
    :param data_column: The name of the column to plot ('Opinion' or 'Bias').
    :param label: The label to use in the plot (e.g., 'Opinion' or 'Bias').
    """
    # Retrieve the agent data from the DataCollector
    agent_data = datacollector.get_agent_vars_dataframe()

    if agent_data.empty:
        print(f"No {label.lower()} data available. Ensure the model has been stepped through multiple iterations.")
        return

    # Ensure the specified data column exists and extract that column
    try:
        data = agent_data[data_column]
    except KeyError:
        print(f"'{data_column}' column not found in the agent data.")
        return

    # Remove rows with None values
    data = data.dropna()

    # Get the number of dimensions by looking at the first agent's data vector
    first_vector = data.iloc[0]
    data_dim = len(first_vector)

    # For each dimension, plot the evolution of values over time for each agent
    for dim in range(data_dim):
        plt.figure(figsize=(10, 6))

        # Get unique Agent IDs
        agent_ids = data.index.get_level_values('AgentID').unique()

        # Iterate over each agent's data in the specified column
        for agent_id in agent_ids:
            # Extract the values for the current dimension over time for this agent
            agent_values_over_time = data.xs(agent_id, level='AgentID').apply(
                lambda values: values[dim]
            )

            # Plot the values for the current agent on the current dimension
            plt.plot(agent_values_over_time.index, agent_values_over_time, label=f'Agent {agent_id}')

        plt.xlabel('Time Step')
        plt.ylabel(f'{label} Value (Dimension {dim + 1})')
        plt.title(f'{label} Evolution Over Time (Dimension {dim + 1})')
        plt.grid(True)
        if data_column != "opinion":
            plt.legend()
        plt.tight_layout()

        # Save and display the plot
        plt.savefig(f'{label.lower()}_evolution_dim_{dim + 1}.png')
        plt.show()


def plot_polarization(datacollector: DataCollector, file_name: str = "polarization.png",
                      title: str = "Polarization Over Time"):
    """
    Plot the polarization over time, as measured by the variance of opinions.
    """
    # Get the recorded polarization data from the DataCollector
    polarization_data = datacollector.get_model_vars_dataframe()["Polarization"]

    # Plotting the polarization over time
    plt.figure(figsize=(10, 6))
    plt.plot(polarization_data, label="Polarization", color="blue")
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Polarization (Opinion Variance)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(file_name)
    plt.show()


def plot_social_network(model):
    """
    Visualize the social network graph where nodes represent agents
    and edges represent connections between them.

    :param model: The NewsMediaModel instance containing the network.
    """
    plt.figure(figsize=(12, 12))

    # Get the network graph from the model
    G = model.grid.G

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

    # Use spring layout for better visualization with a fixed seed for consistency
    pos = nx.spring_layout(G, seed=42)

    # Draw the network
    nx.draw(
        G,
        pos,
        node_color=color_map,
        with_labels=False,
        node_size=300,  # Increase node size for better visibility
        alpha=1.0,  # Ensure nodes are fully opaque
        edge_color='gray',
        linewidths=0.5  # Make edges thinner for better clarity
    )

    # Add a legend to indicate the different types of agents
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='User Agent', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Official News Agent', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Self News Agent', markerfacecolor='red', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Set title and display the graph
    plt.title('Social Network Visualization')
    plt.savefig("graph.png")  # Save the figure
    plt.show()


def plot_opinion_or_bias_frequency(aggregator: RunDataAggregator, metric_type: str, path: str) -> None:
    """
    Plot histograms showing the distribution of opinion or bias strengths in the range [-1, 1]
    for each dimension, overlaying first and last metrics on the same plot.

    :param aggregator: The RunDataAggregator instance containing the aggregated data.
    :param metric_type: The type of metric to plot ('opinion' or 'bias').
    """
    # Get aggregated data based on the selected metric type
    aggregated_data = aggregator.get_aggregated_data()
    if metric_type == 'opinion':
        first_data_all, last_data_all = aggregated_data["first_step_opinions"], aggregated_data["last_step_opinions"]
    elif metric_type == 'bias':
        first_data_all, last_data_all = aggregated_data["first_step_biases"], aggregated_data["last_step_biases"]
    else:
        raise ValueError("Invalid metric type. Choose 'opinion' or 'bias'.")

    # Calculate the averages for the first and last data
    averages_first = calculate_averages(first_data_all)
    averages_last = calculate_averages(last_data_all)
    # Determine all unique dimensions present in the data
    all_dims = {dim for dimensions in averages_first.values() for dim in dimensions}

    # Loop through each dimension and plot
    for dim in all_dims:
        # Extract data for the current dimension
        dim_first = [value[dim] for value in averages_first.values() if dim in value]
        dim_last = [value[dim] for value in averages_last.values() if dim in value]

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.hist(dim_first, bins=np.linspace(-1, 1, 20), alpha=0.5, label=f'First {metric_type.capitalize()} - {dim}')
        plt.hist(dim_last, bins=np.linspace(-1, 1, 20), alpha=0.5, label=f'Last {metric_type.capitalize()} - {dim}')
        plt.title(f'Distribution of {dim} {metric_type.capitalize()}: First vs Last')
        plt.xlim([-1, 1])
        plt.xlabel(f'{metric_type.capitalize()} Strength')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(-1, 1.1, 0.1))
        plt.legend()
        plt.savefig(os.path.join(path, f"{metric_type}_bars_{dim}.svg"))
        plt.show()


def calculate_averages(opinion_data):
    averages = defaultdict(dict)
    for agent_id, dimensions in opinion_data.items():
        for dim, values in dimensions.items():
            averages[agent_id][dim] = np.mean(values)
    return averages
