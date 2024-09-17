from mesa import Agent
from typing import List, Tuple, Any
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import random
from dmas_final_project.agents.user_agent import UserAgent
from dmas_final_project.agents.official_news_agent import OfficialNewsAgent
from dmas_final_project.agents.self_news_agent import SelfNewsAgent


class NewsMediaModel(Model):
    """
    Model representing news media dynamics with user feedback, opinion polarization, and motif-based guidance.
    """

    def __init__(self, num_users: int, num_official_media: int, num_self_media: int, opinion_dims: int,
                 network_type: str, network_params: dict, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the NewsMediaModel.

        :param num_users: Number of user agents.
        :param num_official_media: Number of official news agents.
        :param num_self_media: Number of self-news agents.
        :param opinion_dims: Number of opinion dimensions.
        :param network_type: Type of network ('scale_free', 'small_world', etc.).
        :param network_params: Parameters for generating the network.
        """
        super().__init__(*args, **kwargs)
        self.num_users = num_users
        self.num_official_media = num_official_media
        self.num_self_media = num_self_media
        self.opinion_dims = opinion_dims
        self.network_type = network_type
        self.network_params = network_params
        self.schedule = RandomActivation(self)
        self.running = True

        # Create social network
        self.G = self.create_network()

        # Create agents and place them in the network
        self.create_agents()

        self.datacollector = DataCollector(
            agent_reporters={"Opinion": lambda a: a.opinion if isinstance(a, UserAgent) else None}
        )

    def create_network(self) -> nx.Graph:
        """
        Create a social network graph based on the specified network type.

        :return: A NetworkX graph representing the social network.
        """
        if self.network_type == 'scale_free':
            G = nx.barabasi_albert_graph(self.network_params['n'], self.network_params['m'])
        elif self.network_type == 'small_world':
            G = nx.watts_strogatz_graph(self.network_params['n'], self.network_params['k'], self.network_params['p'])
        else:
            raise ValueError("Unsupported network type specified.")
        return G

    def create_agents(self) -> None:
        """
        Create agents and add them to the network graph.
        """
        for i in range(self.num_users):
            rationality = random.uniform(0.1, 1)
            affective_involvement = random.uniform(0.0, 1)
            #BB: why does rationality start at 0.1 and AffInv at 0.0?
            user = UserAgent(i, self, self.opinion_dims, rationality, affective_involvement)
            self.schedule.add(user)
            self.G.nodes[i]['agent'] = user

        for i in range(self.num_official_media):
            bias = [0] * self.opinion_dims
            official_media = OfficialNewsAgent(i + self.num_users, self, bias)
            self.schedule.add(official_media)
            self.G.nodes[i + self.num_users]['agent'] = official_media

        for i in range(self.num_self_media):
            bias = np.random.uniform(-1, 1, self.opinion_dims)
            self_media = SelfNewsAgent(i + self.num_users + self.num_official_media, self, bias)
            self.schedule.add(self_media)
            self.G.nodes[i + self.num_users + self.num_official_media]['agent'] = self_media

    def get_neighbors(self, agent: Agent) -> List[Agent]:
        """
        Get the neighboring agents of a given agent in the network.

        :param agent: The agent whose neighbors are requested.
        :return: List of neighboring agents.
        """
        node_id = agent.unique_id
        neighbors = [self.G.nodes[n]['agent'] for n in self.G.neighbors(node_id)]
        return neighbors

    def get_user_feedback(self) -> np.ndarray:
        """
        Calculate average user feedback based on opinions to adjust news bias.

        :return: Feedback vector for adjusting news bias.
        """
        opinions = np.array([agent.opinion for agent in self.schedule.agents if isinstance(agent, UserAgent)])
        feedback = np.mean(opinions, axis=0) if len(opinions) > 0 else np.zeros(self.opinion_dims)
        return feedback - 0.5  # Center feedback around 0

    def recognize_motifs(self) -> List[Tuple[UserAgent, UserAgent, UserAgent]]:
        """
        Identify triangular motifs in the network to target for opinion guidance.

        :return: List of triangular motifs with high similarity and rationality.
        """
        motifs = []
        # Only consider UserAgents for motif recognition
        user_agents = [a for a in self.schedule.agents if isinstance(a, UserAgent)]

        for user in user_agents:
            neighbors = [neighbor for neighbor in self.get_neighbors(user) if isinstance(neighbor, UserAgent)]

            if len(neighbors) >= 2:  # Minimum 3 nodes to form a triangle
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        n1 = neighbors[i]
                        n2 = neighbors[j]

                        if self.G.has_edge(n1.unique_id, n2.unique_id):
                            # Calculate similarity between user and the two neighbors
                            similarity = min(user.compute_similarity(n1.opinion), user.compute_similarity(n2.opinion))
                            if similarity > 0.4:  # Threshold for similarity
                                motifs.append((user, n1, n2))
        return motifs

    def apply_guidance(self) -> None:
        """
        Apply opinion guidance to identified motifs by connecting them with diverse agents.
        """
        motifs = self.recognize_motifs()
        for motif in motifs:
            for agent in motif:
                agent.is_guided = True  # Mark agents in motifs as under guidance

    #BB: do we still even want to implement guidance?

    def step(self) -> None:
        """
        Advance the model by one step, apply guidance if necessary, and collect data.
        """
        # Apply motif-based guidance every 30 steps
        if self.schedule.steps % 30 == 0:
            self.apply_guidance()

        self.datacollector.collect(self)
        self.schedule.step()
