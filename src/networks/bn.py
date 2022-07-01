from __future__ import annotations
from src.networks.nodes import DiscreteNode
from typing import Union, Callable
import networkx as nx
import pandas as pd

# Defining types
Id = str
Edge = tuple[Id, Id]


class BayesianNetwork:
    """
    A class for a Bayesian Network.
    """

    def __init__(self):
        # The node map maps the node's ids to the node objects themselves
        self.node_dict: dict[Id, DiscreteNode] = {}
        
        # The graph maps node ids to list of children node ids
        self.graph: dict[Id, list[Id]] = {}
        
        # The node queue lists the topological ordering of the nodes, for inference traversal
        self.node_queue: list[Id] = None
        
    def get_node_queue(self) -> list[Id]:
        return self.node_queue
    
    def get_node_dict(self) -> dict[Id, DiscreteNode]:
        return self.node_dict
    
    def get_graph(self) -> dict[Id, list[Id]]:
        return self.graph

    def draw(self):
        # Create a networkx directed graph and add edges to it
        G = nx.DiGraph(directed=True)
        G.add_edges_from(self.get_edges())
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        
        # Define options for networkx draw method
        options = {
            'node_color': 'orange',
            'node_size': 3000,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        nx.draw_networkx(G, pos, arrows=True, **options)

    def add_nodes(self, nodes: list[DiscreteNode]):
        # Iterate every node to be added
        for node in nodes:
            # Make sure it does not already exist
            if node.get_id() not in self.node_dict:
                self.node_dict[node.get_id()] = node
                self.graph[node.get_id()] = []

    def add_edges(self, edges: list[Edge]):
        # TODO: handle cases where source or destination nodes are not in the Bayesian network.
        # Iterate every edge to be added
        for s, d in edges:
            self.graph[s].append(d)

    def gen_node_queue(self) -> list[Id]:
        """Create the topological node ordering of the Bayesian network using Khan's algorithm.
        This method should only be called once the network structure has been completely defined.

        Returns:
            list[Id]: list of nodes in the Bayesian Network in topological order.
        """
        nodes = [n for n in self.node_dict if self.is_root(n)]
        
        while len(nodes) < len(self.node_dict):
            for node in self.node_dict:
                if node not in nodes:
                    parents = self.get_parents(node)
                    # Add node to list if all its parents are on the list (safe to traverse)
                    if set(parents).issubset(nodes):
                        nodes.append(node)
        return nodes

    def initialize(self):
        self.node_queue = self.gen_node_queue()
        
    def get_node(self, nid: Id):
        return self.node_dict[nid]

    def get_nodes(self) -> list[Id]:
        return self.node_dict.keys()

    def get_edges(self) -> list[Edge]:
        return [(s, d) for s in self.graph for d in self.graph[s]]

    def get_parents(self, node_id: Id) -> list[Id]:
        return [k for k in self.graph if node_id in self.graph[k]]

    def get_pt(self, node_id: Id) -> pd.DataFrame:
        return self.node_dict[node_id].get_pt()
    
    def add_pt(self, node_id: Id, pt: pd.DataFrame):
        self.node_dict[node_id].add_pt(pt)
        
    def change_id(self, old_id: Id, new_id: Id):
        f = lambda x: new_id if x == new_id else x
        for key in self.graph:
            self.graph[key] = list(map(f, self.graph[key]))
        self.graph[new_id] = self.graph.pop(old_id)
        self.node_dict[new_id] = self.node_dict.pop(old_id)
        self.node_dict[new_id].change_id(new_id)
        self.node_queue = list(map(f, self.node_queue)) if self.node_queue is not None else None
        
    def change_ids(self, id_dict: dict[Id, Id]):
        for key in id_dict:
            if key in self.node_dict:
                self.node_dict[id_dict[key]] = self.node_dict.pop(key)
        
    def fix_value(self, node_id: Id, value: int):
        self.node_dict[node_id].fix_value(value)

    def is_leaf(self, node_id: Id) -> bool:
        return len(self.graph[node_id]) == 0

    def is_root(self, node_id: Id) -> bool:
        return len(self.get_parents(node_id)) == 0

    def add_pt(self, node_id: Id, pt: dict[Id, Id]):
        self.node_dict[node_id].add_pt(pt)

    def get_nodes_by_type(self, node_type: str) -> list[Id]:
        return [k for k, v in self.node_dict.items() if v.get_type() == node_type]
    
    def get_nodes_by_key(self, key: Callable[[Id, DiscreteNode], bool]):
        return [n for n in self.node_dict.keys() if key(n, self.node_dict[n])]
    
    def get_sample(self) -> dict[Id, int]:
        """Returns a sample from every node using the direct sampling algorithm. 
        Uses the DiscreteNode class sample method.
        Should only be called after the Bayesian network has been initialized, otherwise returns empty dict.

        Returns:
            dict[Id, int]: a dictionary mapping node ids to their respective sample values.
        """
        # Create empty sample
        sample = {}

        # Sample a result from each node
        for node in self.node_queue:
            sample[node] = self.node_dict[node].get_sample(sample)
            
        return sample

    def query(self, query: list[str], evidence: dict[Id, int] = None, n_samples: int = 100) -> pd.DataFrame:
        """
        Applies the rejection sampling algorithm to approximate any probability distribution.

        Arguments:
            - query ([Id]): node ids for the random variables of the desired probability distribution.
            - evidence ({Id: int}): values for random variables as evidence for the inference. Defaults to None.
            - n_samples (int): number of samples to retrieve. Defaults to 100.

        Return (pd.Dataframe): a dataframe that represents the inferred posterior distribution.
        """
        
        # Initialize evidence as empty dict if it is None
        evidence = {} if evidence is None else evidence
            
        # Create empty DataFrame for sample collection
        sample_df = pd.DataFrame()
        
        # Fix the value of every root node in evidence for faster inference
        root_nodes = [n for n in self.get_nodes() if (self.is_root(n) and n in evidence)]
        backup_pts = {r: self.get_pt(r) for r in root_nodes}
        for r in root_nodes:
            self.fix_value(r, evidence[r])

        # Create multiple samples
        num_samples = 0
        while (num_samples < n_samples):

            # Extract sample from the Bayesian Network
            sample = self.get_sample()

            # Store sample if it matches with evidence
            matches = [sample[name] == evidence[name] for name in evidence]
            if all(matches):
                sample = {k: [v] for k, v in sample.items()}
                sample = pd.DataFrame(sample)
                sample_df = pd.concat([sample_df, sample], ignore_index=True, axis=0)
                num_samples += 1
                
        # Re-change probability tables of root nodes in evidence
        for r in backup_pts:
            self.add_pt(r, backup_pts[r])

        # Turn result into probability table
        sample_df = sample_df.value_counts(normalize=True).to_frame("Prob")

        # Group over query variables and sum over all other variables
        sample_df = sample_df.groupby(query).sum().reset_index()

        return sample_df
