from __future__ import annotations
from src.networks.nodes import Node
from typing import Union
import networkx as nx
import pandas as pd

# Defining types
Id = Union[str, (str, int)]
Edge = tuple[Id, Id]


class BayesianNetwork:
    """
    A class for a Bayesian Network that uses the Node class defined above.
    """

    def __init__(self):
        # The node map maps the node's ids to the node objects themselves
        self.node_map: dict[Id, Node] = {}
        
        # The graph maps node ids to list of children node ids
        self.graph: dict[Id, list[Id]] = {}
        
        # The node queue lists the topological ordering of the nodes, for inference traversal
        self.node_queue: list[Id] = None

    def draw(self):
        # Create a networkx directed graph and add edges to it
        G = nx.DiGraph(directed=True)
        G.add_edges_from(self.get_edges())
        
        # Define options for networkx draw method
        options = {
            'node_color': 'orange',
            'node_size': 3000,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        nx.draw_networkx(G, arrows=True, **options)

    def add_nodes(self, nodes: list[Node]):
        # Iterate every node to be added
        for node in nodes:
            # Make sure it does not already exist
            if node.get_id() not in self.node_map:
                self.node_map[node.get_id()] = node
                self.graph[node.get_id()] = []

    def add_edges(self, edges: list[Edge]):
        # Iterate every edge to be added
        for s, d in edges:
            # Add source node to the bn if it does not already exist
            # if s not in self.graph:
            #    self.add_nodes([s])
            # Add destination node to the bn if it does not already exist
            # if d not in self.graph:
            #    self.add_nodes([d])
            # Add the edge
            self.graph[s].append(d)

    def gen_node_queue(self) -> list[Id]:
        """
        Create the topological node ordering of the Bayesian Network using Khan's algorithm.
        This method should only be called once the network structure has been completely defined.
        """
        nodes = [n for n in self.node_map if self.is_root(n)]
        
        while len(nodes) < len(self.node_map):
            for node in self.node_map:
                if node not in nodes:
                    parents = self.get_parents(node)
                    # Add node to list if all its parents are on the list (safe to traverse)
                    if set(parents).issubset(nodes):
                        nodes.append(node)
        return nodes

    def initialize(self):
        self.node_queue = self.gen_node_queue()
        
    def get_node(self, nid: Id):
        return self.node_map[nid]

    def get_nodes(self) -> list[Id]:
        return self.node_map.keys()

    def get_edges(self) -> list[Edge]:
        edges = []
        for s in self.graph:
            for d in self.graph[s]:
                edges.append((s, d))
        return edges

    def get_parents(self, node_id: Id) -> list[Id]:
        parents = []
        for nid in self.node_map:
            if node_id in self.graph[nid]:
                parents.append(nid)
        return parents

    def get_pt(self, node_id: Id) -> pd.DataFrame:
        return self.node_map[node_id].get_pt()

    def get_node_queue(self) -> list[Id]:
        return self.node_queue

    def is_leaf(self, node_id: Id) -> bool:
        return len(self.graph[node_id]) == 0

    def is_root(self, node_id: Id) -> bool:
        parents = []
        for key in self.graph:
            if node_id in self.graph[key]:
                parents.append(key)
        return len(parents) == 0

    def add_pt(self, node_id: Id, pt: dict[Id, Id]):
        self.node_map[node_id].add_pt(pt)

    def get_nodes_by_type(self, node_type: Type(Node)) -> list[Id]:
        return [k for k, v in self.node_map.items() if type(v) is node_type]

    # FIXME: No mutable arguments as default arguments
    def query(self, query: list[str], evidence: dict[str, int] = {}, n_samples: int = 100) -> pd.DataFrame:
        """
        Applies the rejection sampling algorithm to approximate any probability distribution.

        Arguments:
            - query ([Id]): list of random variables to get the joint distribution from
            - evidence ({Id: int}): dictionary of random variables and their respective values as evidence
            - n_samples (int): number of samples to retrieve

        Return (pd.Dataframe): a dataframe that represents the joint distribution
        """

        # Create empty sampling dictionary
        sample_dict = {name: [] for name in self.node_map}

        # Create multiple samples
        cur_samples = 0
        while (cur_samples < n_samples):

            # Create empty sample and get root nodes
            sample = {}

            # Sample a result from each root node
            for node in self.node_queue:
                # Sample from head of queue
                sample[node] = self.node_map[node].get_sample(sample)

            # Pass sample results to sample_dict if it matches with evidence
            matches = [sample[name] == evidence[name] for name in evidence]
            if all(matches):
                for name in sample_dict:
                    sample_dict[name].append(sample[name])
                cur_samples += 1

        # Turn result into probability table
        df = pd.DataFrame(sample_dict)
        df = df.value_counts(normalize=True).to_frame("Prob")

        # Group over query variables and sum over all other variables
        df = df.groupby(query).sum().reset_index()

        return df
