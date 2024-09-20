from mesa import Agent, Model
import numpy as np

from dmas_final_project.agents.user_agent import UserAgent


class SelfNewsAgent(Agent):
    """
    Represents self-published news that adjusts content bias based on user feedback.
    """

    def __init__(self, unique_id: int, model: Model, bias: np.ndarray, adjustability: float = 0.1) -> None:
        """
        Initialize a SelfNewsAgent.

        :param unique_id: Unique identifier for the agent.
        :param model: The model instance.
        :param bias: Fixed multi-dimensional bias vector of the news.
        :param adjustability: How much the bias can be adjusted.
        """
        super().__init__(unique_id, model)
        self.bias = bias  # Fixed multi-dimensional bias vector
        self.adjustability = adjustability  # Degree of bias adjustment based on feedback

    def step(self) -> None:
        """
        Influence nearby user agents in the network based on the fixed bias.
        """
        # Calculate user feedback to adjust the news bias
        feedback = self.model.get_user_feedback()
        self.bias += self.adjustability * feedback
        # Ensure the bias remains within [-1, 1] bounds
        self.bias = np.clip(self.bias, -1, 1)