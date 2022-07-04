from __future__ import annotations
from src.networks.dn import DecisionNetwork

# Define types
Id = tuple[str, int] # Id's must be tuple for a Dynamic Decision Network
Value = int | float


class DynamicDecisionNetwork(DecisionNetwork):
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
        
        # Get root state nodes
        key = lambda _, node: (node.get_type() == self.state_type) and (node.get_time() == self.get_time())
        query = self.get_nodes_by_key(key)
        
        # Query the next belief-state
        next_belief = self.query(query, {**actions, **observations}, n_samples)
        
        # Change CPT for each root state node
        for nid in query:
            # Get columns to keep in node CPT
            nid_query = list(self.get_node(nid).get_pt().columns)
            
            # Construct CPT by summing over other variables
            b_ = next_belief.groupby(nid_query).sum().reset_index()
            
            # Change cpt of node
            self.get_node(nid).add_pt(b_)