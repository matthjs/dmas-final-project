import numpy as np
from collections import defaultdict


def create_nested_list_defaultdict():
    """Creates a nested defaultdict with list as the innermost value type."""
    return defaultdict(list)


class RunDataAggregator:
    def __init__(self):
        # Use helper function instead of lambda for pickling compatibility
        self.first_step_opinions = defaultdict(create_nested_list_defaultdict)
        self.last_step_opinions = defaultdict(create_nested_list_defaultdict)

        self.first_step_biases = defaultdict(create_nested_list_defaultdict)
        self.last_step_biases = defaultdict(create_nested_list_defaultdict)

    def add_run_data(self, datacollector):
        """
        Extract and store the first and last step opinions from a single run's DataCollector.
        """
        agent_data = datacollector.get_agent_vars_dataframe()

        if agent_data.empty or 'Opinion' not in agent_data.columns:
            return

        # Get the first time step
        time_steps = agent_data.index.get_level_values('Step').unique()
        first_step = time_steps.min()
        last_step = time_steps.max()

        # Store opinions from the first time step
        opinions_first = agent_data.xs(first_step, level='Step')['Opinion'].dropna()
        opinions_last = agent_data.xs(last_step, level='Step')['Opinion'].dropna()

        # Store biases from the first and last time steps
        biases_first = agent_data.xs(first_step, level='Step')['Bias'].dropna()
        biases_last = agent_data.xs(last_step, level='Step')['Bias'].dropna()

        # Track first and last step opinions
        for agent_id, opinion_array in opinions_first.items():
            for dim_index, value in enumerate(opinion_array):
                dim_key = f"dim{dim_index + 1}"
                self.first_step_opinions[agent_id][dim_key].append(value)

        for agent_id, opinion_array in opinions_last.items():
            for dim_index, value in enumerate(opinion_array):
                dim_key = f"dim{dim_index + 1}"
                self.last_step_opinions[agent_id][dim_key].append(value)

        # Track first and last step biases
        for agent_id, bias_array in biases_first.items():
            for dim_index, value in enumerate(bias_array):
                dim_key = f"dim{dim_index + 1}"
                self.first_step_biases[agent_id][dim_key].append(value)

        for agent_id, bias_array in biases_last.items():
            for dim_index, value in enumerate(bias_array):
                dim_key = f"dim{dim_index + 1}"
                self.last_step_biases[agent_id][dim_key].append(value)

    def get_aggregated_data(self):
        """
        Return the aggregated data in dictionary format for plotting.
        """
        return {
            "first_step_opinions": self.first_step_opinions,
            "last_step_opinions": self.last_step_opinions,
            "first_step_biases": self.first_step_biases,
            "last_step_biases": self.last_step_biases
        }
