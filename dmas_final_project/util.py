import matplotlib.pyplot as plt
from dmas_final_project.models.news_media_model import NewsMediaModel

def plot_global_alignment_over_time(model: NewsMediaModel) -> None:
    """
    Plot the global alignment over time.
    
    :param model: The NewsMediaModel instance from which to plot the global alignment.
    """
    # Make sure to run the model for enough steps to collect data
    if not model.global_alignments:
        print("No alignment data available. Ensure the model has been stepped through multiple iterations.")
        return
    
    # Plotting the global alignment over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(model.global_alignments)), model.global_alignments, marker='o', linestyle='-', color='b')
    plt.xlabel('Time Step')
    plt.ylabel('Global Alignment (A(t))')
    plt.title('Global Alignment Over Time')
    plt.grid(True)
    plt.savefig("alignments.png")
    plt.show()
