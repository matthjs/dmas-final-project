import matplotlib.pyplot as plt
from mesa import DataCollector


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
        plt.plot(agent_alignment_data.index.get_level_values('Step'), agent_alignment_data, marker='o', linestyle='',
                 label=f'Agent {agent_id}')

    plt.xlabel('Time Step')
    plt.ylabel('Alignment')
    plt.title('Individual Alignment Over Time')
    plt.grid(True)
    plt.legend()
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
        plt.legend()
        plt.tight_layout()

        # Save and display the plot
        plt.savefig(f'{label.lower()}_evolution_dim_{dim + 1}.png')
        plt.show()
