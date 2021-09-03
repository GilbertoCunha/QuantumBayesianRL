import networkx as nx
import pandas as pd
import numpy as np

class BinaryNode:
    """
    A class for a Bayesian Network node of a boolean random variable
    """
    
    def __init__(self, name):
        self.name = name
        self.parents = []
        self.children = []
        self.pt = None
        
    def add_child(self, name):
        self.children.append(name)
        
    def add_parent(self, name):
        self.parents.append(name)
        
    def add_pt(self, df):
        """
        Adds a probability table to this node
        """
        self.pt = df
        
    def is_root(self):
        return len(self.parents) == 0
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def get_sample(self, sample):
        """
        Samples this node via the direct sampling algorithm
        given previous acquired samples (of ancester nodes)
        """
        
        # Get the row relative to the current sample where current node is false
        sample = {name: sample[name] for name in sample if name in self.parents}
        df = self.pt
        for name in sample:
            df = df.loc[df[name] == sample[name]]
        df = df.loc[df[self.name] == 0]
        
        # Generate random number
        number = np.random.uniform()
        r = int(np.random.uniform() > df["Prob"])
        
        return r


class BayesianNetwork:
    """
    A class for a Bayesian Network that uses the Binary Nodes class defined above.
    """

    def __init__(self):
        self.graph = {}
        self.nodes = []
        self.edges = []
        
    def draw(self):
        G = nx.DiGraph(directed=True)
        G.add_edges_from(self.edges)
        options = {
            'node_color': 'orange',
            'node_size': 3000,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 12,
        }
        nx.draw_networkx(G, arrows=True, **options)
        
    def add_node(self, name):
        if name not in self.graph:
            self.graph[name] = BinaryNode(name)
            self.nodes.append(name)
            
    def add_nodes(self, names):
        for name in names:
            self.add_node(name)
        
    def add_edge(self, origin, dest):
        if (origin in self.graph) and (dest in self.graph):
            self.graph[origin].add_child(dest)
            self.graph[dest].add_parent(origin)
            self.edges.append((origin, dest))
            
    def add_edges(self, edges):
        for o, d in edges:
            self.add_edge(o, d)
            
    def add_node_pt(self, name, df):
        """
        Adds the conditional probability table to a node
        """
        if name in self.graph:
            self.graph[name].add_pt(df)
            
    def query(self, query, evidence, n_samples):
        """
        Applies the direct sampling algorithm
        
        Arguments:
            - query ([str]): list of random variables to get the joint distribution from
            - evidence ({str: int}): dictionary of random variables and their respective values as evidence
            - n_samples (int): number of samples to retrieve
            
        Return (pd.Dataframe): a dataframe that represents the joint distribution
        """
        
        # Create empty sampling dictionary
        sample_dict = {name: [] for name in self.graph}
        
        # Create multiple samples
        cur_samples = 0
        while (cur_samples < n_samples):
            
            # Create empty sample and get root nodes
            sample = {}
            queue = [name for name in self.graph if self.graph[name].is_root()]
            
            # Sample a result from each root node
            while len(queue) != 0:
                
                # Sample from head of queue
                sample[queue[0]] = self.graph[queue[0]].get_sample(sample)
                
                # Add head's children to queue
                queue += self.graph[queue[0]].children
                
                # Remove head from queue
                queue.pop(0)
            
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
        df = df.groupby(query).sum()
        
        return df