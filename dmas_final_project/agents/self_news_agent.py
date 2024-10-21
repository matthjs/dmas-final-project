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

        feedbacks = []
        for neighbor in neighbors:
            if isinstance(neighbor, UserAgent):
                # Calculate cosine similarity between the user's opinion and the news bias
                similarity = np.dot(neighbor.opinion, self.bias) / (
                            np.linalg.norm(neighbor.opinion) * np.linalg.norm(self.bias))

                # Check if similarity exceeds the user-specific threshold
                if abs(similarity) >= neighbor.tolerance_threshold:
                    feedback = similarity  # Use cosine similarity as feedback if above threshold
                else:
                    feedback = 0  # No feedback if similarity is below the threshold

                feedbacks.append(feedback)

        # Return the average feedback from all neighbors (users) or 0 if no neighbors
        return np.mean(feedbacks) if feedbacks else 0

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
