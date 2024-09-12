from mesa import Agent, Model
import numpy as np
from dmas_final_project.agents.user_agent import UserAgent


class OfficialNewsAgent(Agent):
    """
    Represents official news media that adjusts content bias based on user feedback.
    """

    def __init__(self, unique_id: int, model: Model, bias: np.ndarray, adjustability: float = 0.1) -> None:
        """
        Initialize an OfficialNewsAgent.

        :param unique_id: Unique identifier for the agent.
        :param model: The model instance.
        :param bias: Initial multi-dimensional bias vector of the news.
        :param adjustability: How much the bias can be adjusted.
        """
        super().__init__(unique_id, model)
        self.bias = bias  # Multi-dimensional bias vector
        self.adjustability = adjustability  # Degree of bias adjustment based on feedback

    def step(self) -> None:
        """
        Adjust news bias based on aggregated user feedback.
        """
        # Calculate user feedback to adjust the news bias
        feedback = self.model.get_user_feedback()
        self.bias += self.adjustability * feedback
        # Ensure the bias remains within [-1, 1] bounds
        self.bias = np.clip(self.bias, -1, 1)

        # Influence nearby user agents in the network
        neighbors = self.model.get_neighbors(self)
        for neighbor in neighbors:
            if isinstance(neighbor, UserAgent):
                # Influence user's opinion based on the news bias
                influence_factor = neighbor.rationality
                neighbor.opinion += influence_factor * (self.bias - neighbor.opinion)
                # Ensure user's opinion remains within [-1, 1] bounds
                neighbor.opinion = np.clip(neighbor.opinion, -1, 1)
