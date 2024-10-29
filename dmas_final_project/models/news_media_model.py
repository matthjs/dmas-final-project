from mesa import Agent
from typing import List, Tuple, Any, Optional
from mesa import Model
from mesa.space import NetworkGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from dmas_final_project.agents.user_agent import UserAgent
from dmas_final_project.agents.official_news_agent import OfficialNewsAgent
from dmas_final_project.agents.self_news_agent import SelfNewsAgent
import random

from dmas_final_project.data_processing.metrics_tracker import MetricsTracker


class NewsMediaModel(Model):
    """
    Model representing news media dynamics with user feedback, opinion polarization, and motif-based guidance.
    """

    def __init__(self, num_users: int, num_official_media: int, num_self_media: int, opinion_dims: int,
                 network_type: str, network_params: dict, align_freq: int,
                 extra_media_edges: int, extra_self_media_edges: int,
                 enable_feedback: bool = True,
                 seed: Optional[int] = None,
                 user_rationality_mean=0.5, user_rationality_std=0.1,
                 user_affective_involvement_mean=0.5, user_affective_involvement_std=0.1,
                 user_tolerance_threshold_mean=0.5, user_tolerance_threshold_std=0.1,
                 opinion_mean=0, opinion_std=1,  # loc and scale for opinion distribution
                 self_media_bias_mean=0, self_media_bias_std=0.5,
                 self_media_adjustability_mean=0.1, self_media_adjustability_std=0.05,
                 *args: Any, **kwargs: Any) -> None:
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

        self.enable_feedback = enable_feedback

        # Additional graph settings.
        self.extra_media_edges = extra_media_edges
        self.extra_self_media_edges = extra_self_media_edges

        # Normal distribution parameters
        self.opinion_mean = opinion_mean
        self.opinion_std = opinion_std
        self.user_rationality_mean = user_rationality_mean
        self.user_rationality_std = user_rationality_std
        self.user_affective_involvement_mean = user_affective_involvement_mean
        self.user_affective_involvement_std = user_affective_involvement_std
        self.user_tolerance_threshold_mean = user_tolerance_threshold_mean
        self.user_tolerance_threshold_std = user_tolerance_threshold_std
        self.self_media_bias_mean = self_media_bias_mean
        self.self_media_bias_std = self_media_bias_std
        self.self_media_adjustability_mean = self_media_adjustability_mean
        self.self_media_adjustability_std = self_media_adjustability_std

        # Create social network
        self.G = self.create_network()

        self.grid = NetworkGrid(self.G)

        # Create agents and place them in the network
        self.create_agents()

        """
        # Print the number of connections for self-media and official media agents
        for agent in self.schedule.agents:
            if isinstance(agent, SelfNewsAgent):
                num_connections = len(self.G[agent.unique_id])
                print(f"Self-media agent {agent.unique_id} has {num_connections} connections.")
            elif isinstance(agent, OfficialNewsAgent):
                num_connections = len(self.G[agent.unique_id])
                print(f"Official media agent {agent.unique_id} has {num_connections} connections.")
        """

        # Initialize the MetricsTracker for global alignment and polarization
        self.metrics_tracker = MetricsTracker()
        # Record the metrics in the MetricsTracker
        self.metrics_tracker.register_metric('Global Alignment')
        self.metrics_tracker.register_metric('Polarization')
        self.metrics_tracker.register_metric('Homophily Index')  # New metric added
        self.metrics_tracker.register_metric('Mean Opinion Magnitude')  # New metric added

        # DataCollector to track global alignment
        self.datacollector = DataCollector(
            model_reporters={
                "Global Alignment": lambda m: m.global_alignment if m.schedule.steps % m.align_freq == 0 else None,
                "Polarization": lambda m: m.compute_polarization(),
                "Homophily Index": lambda m: m.compute_homophily_index(
                    threshold=0.5) if m.schedule.steps % m.align_freq == 0 else None,  # Compute only at align_freq
                "Mean Opinion Magnitude": lambda
                    m: m.compute_magnitude_mean_opinion()
            },
            agent_reporters={"Opinion": lambda a: a.opinion if isinstance(a, UserAgent) else None,
                             "Bias": lambda a: a.bias if isinstance(a, SelfNewsAgent) else None,
                             "Alignment":
                                 lambda a: a.alignment if isinstance(a,
                                                                     UserAgent) and self.schedule.steps % self.align_freq == 0 else None}
        )

        self.global_alignment = None
        self.principal_components = None

    def set_metrics_tracker(self, metric_tracker):
        self.metrics_tracker = metric_tracker

    def compute_polarization(self) -> float:
        """
        Compute the polarization of opinions among user agents in the model.
        Polarization is measured as the variance of the opinion vectors across all user agents.
        """
        user_opinions = [agent.opinion for agent in self.schedule.agents if isinstance(agent, UserAgent)]
        if not user_opinions:
            return 0  # Return 0 if there are no user agents

        opinions_matrix = np.array(user_opinions)
        # Calculate variance across all opinion dimensions
        polarization = np.mean(np.var(opinions_matrix, axis=0))
        return polarization

    def compute_magnitude_mean_opinion(self) -> float:
        """
        Compute the scalar mean opinion as the magnitude of the mean opinion vector across all user agents.
        This represents the strength of consensus among the population.
        """
        user_opinions = [agent.opinion for agent in self.schedule.agents if isinstance(agent, UserAgent)]
        if not user_opinions:
            return 0.0  # Return 0 if there are no user agents

        # Calculate the mean opinion vector
        opinions_matrix = np.array(user_opinions)
        mean_opinion_vector = np.mean(opinions_matrix, axis=0)

        # Compute the magnitude (Euclidean norm) of the mean opinion vector
        mean_opinion_magnitude = np.linalg.norm(mean_opinion_vector)

        return mean_opinion_magnitude

    def compute_homophily_index(self, threshold=1) -> float:
        """
        Compute the homophily index for user agents based on a cosine similarity threshold.
        :param threshold: The similarity threshold for counting edges between similar user agents.
        :return: The homophily index for user agents.
        """
        user_agents = [agent for agent in self.schedule.agents if isinstance(agent, UserAgent)]
        user_edges = [(i, j) for i in user_agents for j in self.grid.get_neighbors(i.pos, include_center=False) if isinstance(j, UserAgent)]

        if len(user_edges) == 0:
            return 0  # No edges between user agents

        similar_edges = 0

        for i, j in user_edges:
            # Compute cosine similarity between opinion vectors of user agents i and j
            similarity = i.compute_similarity(j.opinion)
            if similarity >= threshold:
                similar_edges += 1

        # Homophily Index H_rho
        homophily_index = similar_edges / len(user_edges) if user_edges else 0
        return homophily_index

    def create_network(self) -> nx.Graph:
        """
        Create a social network graph based on the specified network type.

        :return: A NetworkX graph representing the social network.
        """
        if self.network_type == 'scale_free':
            n = self.num_users + self.num_self_media + self.num_official_media
            G = nx.barabasi_albert_graph(n, self.network_params['m'])
        elif self.network_type == 'small_world':
            n = self.num_users + self.num_self_media + self.num_official_media
            G = nx.watts_strogatz_graph(n, self.network_params['k'], self.network_params['p'])
        else:
            raise ValueError("Unsupported network type specified.")

        # Add extra edges between media agents and user agents
        user_nodes = range(self.num_users)  # IDs for user agents
        official_media_nodes = range(self.num_users,
                                     self.num_users + self.num_official_media)  # IDs for official media agents
        self_media_nodes = range(self.num_users + self.num_official_media, n)  # IDs for self media agents

        # Add more connections between official media and user agents
        for media_id in official_media_nodes:
            num_extra_edges = self.extra_media_edges
            connected_users = np.random.choice(user_nodes, num_extra_edges, replace=False)
            for user_id in connected_users:
                G.add_edge(media_id, user_id)

        # Add more connections between self-media and user agents
        for media_id in self_media_nodes:
            num_extra_edges = self.extra_self_media_edges
            connected_users = np.random.choice(user_nodes, num_extra_edges, replace=False)
            for user_id in connected_users:
                G.add_edge(media_id, user_id)

        return G

    def create_agents(self):
        agent_id = 0
        # Create user agents with parameters drawn from normal distributions
        for i in range(self.num_users):
            opinion = np.random.normal(self.opinion_mean, self.opinion_std, self.opinion_dims)
            opinion = np.clip(opinion, -1, 1)  # Ensure opinions are between -1 and 1

            rationality = np.random.normal(self.user_rationality_mean, self.user_rationality_std)
            affective_involvement = np.random.normal(self.user_affective_involvement_mean,
                                                     self.user_affective_involvement_std)
            tolerance_threshold = np.random.normal(self.user_tolerance_threshold_mean,
                                                   self.user_tolerance_threshold_std)
            # Ensure values are within the desired range (e.g., [0, 1])
            rationality = np.clip(rationality, 0, 1)
            affective_involvement = np.clip(affective_involvement, 0, 1)
            tolerance_threshold = np.clip(tolerance_threshold, 0, 1)

            user = UserAgent(agent_id, self, opinion, rationality, affective_involvement, tolerance_threshold)
            self.schedule.add(user)
            self.grid.place_agent(user, agent_id)
            agent_id += 1

        # Create official media agents (you can add more customization if needed)
        for i in range(self.num_official_media):
            official_media = OfficialNewsAgent(agent_id, self, self.opinion_dims)
            self.schedule.add(official_media)
            self.grid.place_agent(official_media, agent_id)
            agent_id += 1

        # Create self-media agents with bias and adjustability drawn from normal distributions
        for i in range(self.num_self_media):
            bias = np.random.normal(self.self_media_bias_mean, self.self_media_bias_std, self.opinion_dims)
            adjustability = np.random.normal(self.self_media_adjustability_mean, self.self_media_adjustability_std)
            # Ensure adjustability is within a sensible range
            adjustability = np.clip(adjustability, 0, 1)

            self_media = SelfNewsAgent(agent_id, self, bias, adjustability,
                                       enable_feedback=self.enable_feedback)
            self.schedule.add(self_media)
            self.grid.place_agent(self_media, agent_id)
            agent_id += 1

    def compute_alignments(self):
        try:
            component, global_alignment = self.compute_global_alignment()
            self.principal_components, self.global_alignment = component, global_alignment
        except ValueError:
            logger.warning("Nan opinion vectors in opinion space, skipping PCA.")
            return

        user_agents = [agent for agent in self.schedule.agents if isinstance(agent, UserAgent)]
        for user_agent in user_agents:
            user_agent.compute_alignment(self.principal_components[0])  # Individual alignment is computed
            # based on first principle component.

    def compute_global_alignment(self):
        # NOTE: The opinion space now contains nan vectors this needs to be fixed.
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

    def step(self):
        """
        Advance the model by one step, apply guidance if necessary, and collect data.
        """
        self.schedule.step()

        if self.schedule.steps % self.align_freq == 0:
            self.compute_alignments()

        # Collect global alignment and polarization
        global_alignment_data = self.datacollector.get_model_vars_dataframe()
        self.datacollector.collect(self)

        if not global_alignment_data.empty:
            most_recent_entry = global_alignment_data.iloc[-1]
            global_alignment = most_recent_entry.get("Global Alignment")
            polarization = most_recent_entry.get("Polarization")
            homophily_index = most_recent_entry.get("Homophily Index")
            scalar_mean_opinion = most_recent_entry.get('Mean Opinion Magnitude')

            if global_alignment is not None:
                self.metrics_tracker.record_metric('Global Alignment', 'model', self.schedule.steps, global_alignment)

            if polarization is not None:
                self.metrics_tracker.record_metric('Polarization', 'model', self.schedule.steps, polarization)

            if homophily_index is not None:
                self.metrics_tracker.record_metric('Homophily Index', 'model', self.schedule.steps, homophily_index)

            if scalar_mean_opinion is not None:
                self.metrics_tracker.record_metric('Mean Opinion Magnitude', 'model', self.schedule.steps, scalar_mean_opinion)  # New metric recorded