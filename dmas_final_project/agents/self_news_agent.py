from mesa import Agent, Model
import numpy as np

from dmas_final_project.agents.user_agent import UserAgent


class SelfNewsAgent(Agent):
    """
    Represents self-published news with a fixed bias.
    """

    def __init__(self, unique_id: int, model: Model, bias: np.ndarray) -> None:
        """
        Initialize a SelfNewsAgent.

        :param unique_id: Unique identifier for the agent.
        :param model: The model instance.
        :param bias: Fixed multi-dimensional bias vector of the news.
        """
        super().__init__(unique_id, model)
        self.bias = bias  # Fixed multi-dimensional bias vector

    def step(self) -> None:
        """
        Influence nearby user agents in the network based on the fixed bias.
        """
        # Influence nearby user agents in the network
        neighbors = self.model.get_neighbors(self)
        for neighbor in neighbors:
            if isinstance(neighbor, UserAgent):
                # Influence user's opinion based on the news bias
                influence_factor = neighbor.rationality
                neighbor.opinion += influence_factor * (self.bias - neighbor.opinion)
                # Ensure user's opinion remains within [-1, 1] bounds
                neighbor.opinion = np.clip(neighbor.opinion, -1, 1)