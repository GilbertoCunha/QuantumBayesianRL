from __future__ import annotations
from src.networks.nodes import StaticNode, StaticActionNode, StaticUtilityNode
from src.networks.qbn import QuantumBayesianNetwork
from src.networks.bn import BayesianNetwork
import matplotlib.pyplot as plt
from typing import Type, Union
import networkx as nx
import pandas as pd
import itertools

# Defining types
Node = Union[StaticNode, StaticActionNode, StaticUtilityNode]

class DecisionNetwork(BayesianNetwork):
    
    def __init__(self):
        super().__init__()
            
    def draw(self):
        # Create nx graph
        G = nx.DiGraph(directed=True)
        G.add_edges_from(self.get_edges())
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        
        # Draw regular nodes
        nodes = self.get_nodes_by_type(StaticNode)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color="orange", node_size=3000, node_shape="o")
        
        # Draw action nodes
        action_nodes = self.get_nodes_by_type(StaticActionNode)
        nx.draw_networkx_nodes(G, pos, nodelist=action_nodes, node_color="blue", node_size=3000, node_shape="s")
        
        # Draw action nodes
        utility_nodes = self.get_nodes_by_type(StaticUtilityNode)
        nx.draw_networkx_nodes(G, pos, nodelist=utility_nodes, node_color="green", node_size=3000, node_shape="d")
        
        # Draw network edges
        nx.draw_networkx_edges(G, pos, node_size=3000)
        
        # Draw node labels
        labels = {n: n for n in self.node_map}
        nx.draw_networkx_labels(G, pos, labels)
        plt.show()
        
    @staticmethod
    def bitGen(n: int):
        return [''.join(i) for i in itertools.product('01', repeat=n)]
    
    def query_decision(self, query: list[str], evidence: dict[str, int], n_samples: int = 1000, quantum: bool = False) -> dict[str, int]:
        # Get all action nodes
        action_nodes = self.get_nodes_by_type(StaticActionNode)
        
        # Get all actions for all the action nodes
        action_space = {}
        for a in action_nodes:
            start, stop = self.node_map[a].get_value_range()
            action_space[a] = list(range(start, stop+1))
        
        # Create a list of all possible actions to be taken
        keys, values = zip(*action_space.items())
        action_space = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Iterate each set of actions in action space
        results = []
        for actions in action_space:
            
            # Set the actions of the action nodes to the current set of actions
            for action_node in action_nodes:
                self.node_map[action_node].add_value(actions[action_node])
                
            # Perform query
            if quantum:
                qbn = QuantumBayesianNetwork(self)
                df = qbn.query(query=[query], evidence=evidence, n_samples=n_samples)
            else:
                df = self.query(query=[query], evidence=evidence, n_samples=n_samples)
            
            # Get expected utility
            eu = float((df[query] * df["Prob"]).sum())
            results.append((actions, eu))
            
        # Get the result with the maximum expected utility
        r = max(results, key=lambda x: x[1])[0]
        
        return r
        
