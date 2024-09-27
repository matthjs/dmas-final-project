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