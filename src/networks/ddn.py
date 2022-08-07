from __future__ import annotations
from src.networks.qbn import QuantumBayesianNetwork
from src.networks.bn import BayesianNetwork
from src.networks.dn import DNFactory
from src.utils import belief_update
from typing import Union
import pandas as pd

# Define types
Id = tuple[str, int] # Id's must be tuple for a Dynamic Decision Network
Value = Union[int, float]
SpaceElement = dict[Id, Value]
Space = list[SpaceElement]
BeliefState = dict[Id, pd.DataFrame]
Network = Union[BayesianNetwork, QuantumBayesianNetwork]


def DDNFactory(Base: Network, discount: float = 1.0):
    
    class DynamicDecisionNetwork(DNFactory(Base)):
        """Extends the decision network class to implement a dynamic decision network.
        These dynamic decision networks can model POMDPs.
        """
        
        # TODO: Add current state of environment (the agent does not know it, but it is needed for sampling observations)
        
        def __init__(self, discount: float = 1.0):
            super().__init__()
            self.discount = discount
            self.time: int = None
            
        def get_discount(self) -> float:
            return self.discount
        
        def get_time(self) -> int:
            if self.time is None:
                r = min([self.get_node(nid).get_time() for nid in self.get_nodes()])
            else:
                r = self.time
            return r
        
        def get_space(self, node_type: str) -> Space:
            r = {}
            nodes = self.get_nodes_by_type(node_type)
            for node in nodes:
                r[node] = self.get_node(node).get_value_space()
            return r
        
        def get_belief_state(self) -> BeliefState:
            r = {}
            for node in self.get_root_state_nodes():
                r[node] = self.get_node(node).get_pt()
            return r
        
        def get_root_state_nodes(self) -> list[Id]:
            key = lambda _, node: (node.get_type() == self.state_type) and (node.get_time() == self.get_time())
            return self.get_nodes_by_key(key)
        
        def initialize(self):
            super().initialize()
            self.time = self.get_time()
            
        def increase_time(self):
            # TODO: throw exception when id type is not tuple
            
            # Iterate nodes in reverse order (due to time-step increase for no key collisions)
            for n, t in self.node_queue[::-1]:
                self.node_dict[(n, t)].increase_time()
                self.node_dict[(n, t+1)] = self.node_dict.pop((n, t))
                self.graph[(n, t+1)] = self.graph.pop((n, t))
                self.graph[(n, t+1)] = [(n_, t_+1) for (n_, t_) in self.graph[(n, t+1)]]
            
            # Change node queue
            self.node_queue = [(n, t+1) for (n, t) in self.node_queue]
            self.time += 1
            
        def sample_observation(self, actions: dict[Id, Value]) -> dict[Id, Value]:
            # TODO: check if actions dict is correct
            
            query = self.get_nodes_by_type(self.observation_type)
            sample = self.query(query, actions, n_samples=1)[query].to_dict(orient="list")
            sample = {k: v.pop() for k, v in sample.items()}
            return sample
        
        def belief_update(self, actions: dict[Id, Value], observations: dict[Id, Value], n_samples: int = 100):
            # TODO: check if actions and observations dict are correct
            
            # Get new belief-state
            new_belief = belief_update(self, self.get_belief_state(), actions, observations, n_samples)
            
            # Update CPTs for the new-belief
            for node in new_belief:
                self.get_node(node).add_pt(new_belief[node])
                
    return DynamicDecisionNetwork