from __future__ import annotations
from src.networks.nodes import Node
import networkx as nx
import pandas as pd
import numpy as np

# Defining types
Edge = (str, str)


class BayesianNetwork:
    """
    A class for a Bayesian Network that uses the Binary Nodes class defined above.
    """

    def __init__(self):
        self.node_map: dict[str,Node] = {}
        self.graph: dict[str,list[str]] = {}

    def draw(self):
        G = nx.DiGraph(directed=True)
        G.add_edges_from(self.get_edges())
        options = {
            'node_color': 'orange',
            'node_size': 3000,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        nx.draw_networkx(G, arrows=True, **options)

    def add_nodes(self, nodes: list[Node]):
        for node in nodes:
            if node.get_id() not in self.node_map:
                self.node_map[node.get_id()] = node
                self.graph[node.get_id()] = []

    def add_edges(self, edges: list[Edge]):
        for edge in edges:
            s, d = edge
            if s not in self.graph:
                self.graph[s] = []
            self.graph[s].append(d)

    def gen_node_queue(self):
        nodes = [n for n in self.node_map if self.is_root(n)]
        while len(nodes) < len(self.node_map):
            for node in self.node_map:
                if node not in nodes:
                    parents = self.get_parents(node)
                    if set(parents).issubset(nodes):
                        nodes.append(node)
        self.node_queue = nodes

    def initialize(self):
        self.gen_node_queue()

    def get_nodes(self) -> list[str]:
        return self.node_map.keys()

    def get_edges(self) -> list[(str, str)]:
        edges = []
        for s in self.graph:
            for d in self.graph[s]:
                edges.append((s, d))
        return edges

    def get_parents(self, node_id: str) -> list[str]:
        parents = []
        for nid in self.node_map:
            if node_id in self.graph[nid]:
                parents.append(nid)
        return parents

    def get_pt(self, node_id: str) -> pd.DataFrame:
        return self.node_map[node_id].get_pt()

    def get_node_queue(self) -> list[str]:
        return self.node_queue

    def is_leaf(self, node_id: str) -> bool:
        # For a node to be leaf it cant have children
        return len(self.graph[node_id]) == 0

    def is_root(self, node_id: str) -> bool:
        # For a node to be root it cant have parents
        parents = []
        for key in self.graph:
            if node_id in self.graph[key]:
                parents.append(key)
        return len(parents) == 0

    def add_pt(self, node_id: str, pt: dict[str, int]):
        self.node_map[node_id].add_pt(pt)

    def get_nodes_by_type(self, node_type: Type(Node)) -> list[str]:
        return [k for k, v in self.node_map.items() if type(v) is node_type]

    def query(self, query: list[str], evidence: dict[str, int] = {}, n_samples: int = 1000) -> pd.DataFrame:
        """
        Applies the direct sampling algorithm

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
