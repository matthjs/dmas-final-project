from mesa import Agent, Model
import numpy as np

from dmas_final_project.agents.user_agent import UserAgent


class SelfNewsAgent(Agent):
    """
    Represents self-published news that adjusts content bias based on user feedback.
    """

    def __init__(self, unique_id: int, model: Model, bias: np.ndarray, adjustability: float = 0.1,
                 enable_feedback: bool = True) -> None:
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
        self.enable_feedback = enable_feedback

    def get_user_feedback(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)

        dissonances = []
        opinion_dim = len(self.bias)  # Assuming self.bias is an M-dimensional vector
        for neighbor in neighbors:
            if isinstance(neighbor, UserAgent):
                # Scale dissonance by sqrt(M) to avoid large feedback values in high dimensions
                dissonance = (-np.linalg.norm(neighbor.opinion - self.bias) /
                              np.sqrt(opinion_dim) + neighbor.tolerance_threshold)
                dissonances.append(dissonance)

        feedback = np.mean(dissonances) if dissonances else 0  # Handle case with no neighbors
        return feedback

    def step(self) -> None:
        """
        Influence nearby user agents in the network based on the fixed bias.
        """
        if self.enable_feedback:
            # Calculate user feedback to adjust the news bias
            feedback = self.get_user_feedback()
            self.bias += self.adjustability * feedback * self.bias
            # Ensure the bias remains within [-1, 1] bounds
            self.bias = np.clip(self.bias, -1, 1)

        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        for neighbor in neighbors:
            if isinstance(neighbor, UserAgent):
                neighbor.opinion += neighbor.rationality * (self.bias - neighbor.opinion)
                neighbor.opinion = np.clip(neighbor.opinion, -1, 1)
