from mesa import Agent, Model
import numpy as np


class UserAgent(Agent):
    """
    Represents a user with multi-dimensional opinions in a social network.
    Opinions change based on cognitive dissonance.
    """

    def __init__(self, unique_id: int, model: Model, opinion: np.ndarray, rationality: float,
                 affective_involvement: float, tolerance_threshold: float) -> None:
        """
        Initialize a UserAgent.

        :param unique_id: Unique identifier for the agent.
        :param model: The model instance.
        :param opinion: Multi-dimensional opinion vector.
        :param rationality: Level of rationality affecting opinion change speed.
        :param affective_involvement: Emotional involvement level affecting opinion resistance.
        :param tolerance_threshold: Tolerance level for cognitive dissonance (news feedback) before opinion changes.
        """
        super().__init__(unique_id, model)
        self.opinion = opinion  # Multi-dimensional opinion vector passed as a parameter
        self.rationality = rationality  # Affects how opinions are updated
        self.affective_involvement = affective_involvement  # Affects resistance to opinion change; the higher, the less change
        self.tolerance_threshold = tolerance_threshold
        self.is_guided = False  # Indicates if the agent is under opinion guidance
        self.alignment = None

    def compute_alignment(self, principal_component: np.ndarray) -> None:
        # Calculate the angle phi^i(t) between the opinion vector and the first principal component
        cos_phi = self.compute_similarity(principal_component)

        # Compute the angle phi^i(t)
        angle = np.arccos(cos_phi)

        # Compute the individual alignment a^i(t)
        self.alignment = np.abs((2 * angle / np.pi) - 1)

    def compute_similarity(self, other_opinion: np.ndarray) -> float:
        """
        Compute directional similarity with another agent.

        :param other_opinion: Opinion vector of another agent.
        :return: Directional similarity between the two opinion vectors.
        """
        dot_product = np.dot(self.opinion, other_opinion)
        norm_product = np.linalg.norm(self.opinion) * np.linalg.norm(other_opinion)
        similarity = dot_product / norm_product if norm_product != 0 else 0
        return similarity

    def step(self) -> None:
        """
        Update agent's opinion based on interactions with neighboring agents.
        """
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        for neighbor in neighbors:
            if isinstance(neighbor, UserAgent):
                similarity = self.compute_similarity(neighbor.opinion)
                if similarity > self.affective_involvement:  # Similar enough to interact
                    # Cognitive dissonance adjustment
                    self.opinion += self.rationality * (neighbor.opinion - self.opinion)

        # Normalize opinion to [-1, 1] range
        self.opinion = np.clip(self.opinion, -1, 1)
