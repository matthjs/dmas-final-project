from mesa import Agent, Model
import numpy as np
from dmas_final_project.agents.user_agent import UserAgent


class OfficialNewsAgent(Agent):
    """
    Represents official news media with fixed bias.
    """

    def __init__(self, unique_id: int, model: Model, bias: np.ndarray) -> None:
        """
        Initialize an OfficialNewsAgent.

        :param unique_id: Unique identifier for the agent.
        :param model: The model instance.
        :param bias: Initial multi-dimensional bias vector of the news.
        """
        super().__init__(unique_id, model)
        self.bias = bias  # Multi-dimensional bias vector

    def step(self) -> None:
        """
        Adjust news bias based on aggregated user feedback.
        """
        pass
