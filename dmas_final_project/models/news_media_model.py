from mesa import Agent
from typing import List, Tuple, Any
from mesa import Model
from mesa.space import NetworkGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import random
from sklearn.decomposition import PCA
from dmas_final_project.agents.user_agent import UserAgent
from dmas_final_project.agents.official_news_agent import OfficialNewsAgent
from dmas_final_project.agents.self_news_agent import SelfNewsAgent


class NewsMediaModel(Model):
    """
    Model representing news media dynamics with user feedback, opinion polarization, and motif-based guidance.
    """

    def __init__(self, num_users: int, num_official_media: int, num_self_media: int, opinion_dims: int,
                 network_type: str, network_params: dict, align_freq: int, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the NewsMediaModel.

        :param num_users: Number of user agents.
        :param num_official_media: Number of official news agents.
        :param num_self_media: Number of self-news agents.
        :param opinion_dims: Number of opinion dimensions.
        :param network_type: Type of network ('scale_free', 'small_world', etc.).
        :param network_params: Parameters for generating the network.
        :param align_freq: how 
        """
        super().__init__(*args, **kwargs)
        self.num_users = num_users
        self.num_official_media = num_official_media
        self.num_self_media = num_self_media
        self.opinion_dims = opinion_dims
        self.network_type = network_type
        self.network_params = network_params
        self.align_freq = align_freq
        self.schedule = RandomActivation(self)
        self.running = True

        # Create social network
        self.G = self.create_network()

        self.grid = NetworkGrid(self.G)

        # Create agents and place them in the network
        self.create_agents()

        self.datacollector = DataCollector(
            agent_reporters={"Opinion": lambda a: a.opinion if isinstance(a, UserAgent) else None}
        )

        self.global_alignment = None
        self.global_alignments = []
        self.principal_components = None

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

    def create_agents(self):
        agent_id = 0
        for i in range(self.num_users):
            rationality = self.random.uniform(0.1, 1)
            affective_involvement = self.random.uniform(0.2, 1)
            tolerance_threshold = self.random.uniform(0.1, 1)
            user = UserAgent(agent_id, self, self.opinion_dims, rationality, affective_involvement,
                             tolerance_threshold)
            self.schedule.add(user)
            self.grid.place_agent(user, agent_id)
            agent_id += 1

        for i in range(self.num_official_media):
            official_media = OfficialNewsAgent(agent_id, self, self.opinion_dims)
            self.schedule.add(official_media)
            self.grid.place_agent(official_media, agent_id)
            agent_id += 1

        for i in range(self.num_self_media):
            bias = np.random.uniform(-1, 1, self.opinion_dims)
            adjustability = 0.1
            self_media = SelfNewsAgent(agent_id, self, bias, adjustability)
            self.schedule.add(self_media)
            self.grid.place_agent(self_media, agent_id)
            agent_id += 1

    def compute_alignments(self):
        self.principal_components, self.global_alignment = self.compute_global_alignment()
        self.global_alignments.append(self.global_alignment)

        user_agents = [agent for agent in self.schedule.agents if isinstance(agent, UserAgent)]
        for user_agent in user_agents:
            user_agent.compute_alignment(self.principal_components[0])     # Individual alignment is computed
            # based on first principle component.

    def compute_global_alignment(self):
        # Double check this.
        opinion_space = np.array(
            [agent.opinion for agent in self.schedule.agents if isinstance(agent, UserAgent)])
        
        # Perform PCA
        pca = PCA(n_components=self.opinion_dims)
        principal_components = pca.fit_transform(opinion_space)

        # First Principal Component (c1)
        c1 = pca.components_[0]

        # Variance explained by the first principal component (λ1)
        lambda_1 = pca.explained_variance_[0]

        # A(t) = λ1 (the variance explained by the first principal component)
        A_t = lambda_1

        # print("First Principal Component (c1):", c1)
        # print("Variance Explained by c1 (A(t) = λ1):", A_t)

        return principal_components, A_t
        

    def get_neighbors(self, agent: Agent) -> List[Agent]:
        """
        Get the neighboring agents of a given agent in the network.

        :param agent: The agent whose neighbors are requested.
        :return: List of neighboring agents.
        """
        node_id = agent.unique_id
        neighbors = [self.G.nodes[n]['agent'] for n in self.G.neighbors(node_id)]
        return neighbors

    def recognize_motifs(self) -> List[Tuple[UserAgent, UserAgent, UserAgent]]:
        """
        Identify triangular motifs in the network to target for opinion guidance.

        :return: List of triangular motifs with high similarity and rationality.
        """
        # ML: This may need to be adjusted.
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
        self.schedule.step()

        # Apply motif-based guidance every 30 steps
        if self.schedule.steps % 30 == 0:
            self.apply_guidance()

        if self.schedule.steps % self.align_freq:
            self.compute_alignments()

        self.datacollector.collect(self)

