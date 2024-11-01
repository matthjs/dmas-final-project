from mesa import Agent, Model
import numpy as np
from dmas_final_project.agents.user_agent import UserAgent


class OfficialNewsAgent(Agent):
    """
    Represents official news media with fixed bias.
    """

    def __init__(self, unique_id: int, model: Model, opinion_dims: int) -> None:
        """
        Initialize an OfficialNewsAgent.

        :param unique_id: Unique identifier for the agent.
        :param model: The model instance.
        :param opinion_dims: Dimensionality of the opinion space.
        """
        super().__init__(unique_id, model)
        self.news = np.zeros(opinion_dims)

    def step(self) -> None:
        """
        Push 'news vector' to
        """
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        for neighbor in neighbors:
            if isinstance(neighbor, UserAgent):
                neighbor.opinion += neighbor.rationality * (self.news - neighbor.opinion)
                neighbor.opinion = np.clip(neighbor.opinion, -1, 1)

