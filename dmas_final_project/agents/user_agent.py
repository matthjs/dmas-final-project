from mesa import Agent, Model
import numpy as np


class UserAgent(Agent):
    """
    Represents a user with multi-dimensional opinions in a social network.
    Opinions change based on cognitive dissonance and motif-guided interventions.
    """

    def __init__(self, unique_id: int, model: Model, opinion_dims: int, rationality: float,
                 affective_involvement: float, tolerance_threshold: float) -> None:
        """
        Initialize a UserAgent.

        :param unique_id: Unique identifier for the agent.
        :param model: The model instance.
        :param opinion_dims: Number of opinion dimensions.
        :param rationality: Level of rationality affecting opinion change speed.
        :param affective_involvement: Emotional involvement level affecting opinion resistance.
        """
        super().__init__(unique_id, model)
        self.opinion = np.random.uniform(-1, 1, opinion_dims)  # Multi-dimensional opinion vector
        self.rationality = rationality  # Affects how opinions are updated
        self.affective_involvement = affective_involvement  # Affects resistance to opinion change the higher, the less change
        self.tolerance_threshold = tolerance_threshold
        self.is_guided = False  # Indicates if the agent is under opinion guidance
        self.alignment = None

    def compute_alignment(self, principal_component: np.ndarray) -> None:
        # Calculate the angle phi^i(t) between the opinion vector and the first principal component
        dot_product = np.dot(self.opinion, principal_component)
        norm_opinion = np.linalg.norm(self.opinion)
        norm_principal = np.linalg.norm(principal_component)
        cos_phi = dot_product / (norm_opinion * norm_principal)

        # Ensure the value is within the valid range for arccos due to floating-point precision issues
        cos_phi = np.clip(cos_phi, -1.0, 1.0)

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
        return max(0, similarity) 


    def step(self) -> None:
        """
        Update agent's opinion based on interactions with neighboring agents.
        """
        neighbors = self.model.get_neighbors(self)
        for neighbor in neighbors:
            if hasattr(neighbor, 'bias'):  # Check if the neighbor has a 'bias' attribute
                # Update opinion based on news bias and rationality
                self.opinion += self.rationality * (neighbor.bias - self.opinion)
                #BB: will we add a similarity treshold for media? if so, maybe if statements are unnecessary, same method for everything
            elif isinstance(neighbor, UserAgent):
                similarity = self.compute_similarity(neighbor.opinion)
                if similarity > self.affective_involvement:  # Similar enough to interact
                    # Cognitive dissonance adjustment
                    self.opinion += self.rationality * (neighbor.opinion - self.opinion)

        # Normalize opinion to [-1, 1] range
        self.opinion = np.clip(self.opinion, -1, 1)
