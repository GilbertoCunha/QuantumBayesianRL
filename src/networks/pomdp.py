from __future__ import annotations
from src.networks.dn import DecisionNetwork

# Define types
Id = str | tuple[str, int]
Value = int


class POMDP(DecisionNetwork):
    
    def __init__(self, discount: float):
        super().__init__()
        self.discount: float = discount
        
    def get_discount(self) -> float:
        return self.discount
        
    def query_decision(self, *args, **kwargs):
        raise NotImplemented("Method query_decision not implemented for POMDP.")
        
    def get_observation_sample(self, actions: dict[Id, Value]) -> dict[Id, Value]:
        query = self.get_nodes_by_type(self.observation_type)
        sample = self.query(query, actions, n_samples=1).to_dict(orient="list")
        sample = {k: v.pop() for k, v in sample.items()}
        return sample
    
    def belief_update(self, actions: dict[Id, Value], observations: dict[Id, Value], n_samples: int = 100):
        # Get root state nodes
        key = lambda _, node: (node.get_type() == self.state_type) and (node.get_time() == self.get_time())
        query = self.get_nodes_by_key(key)
        
        # Query the next belief-state
        next_belief = self.query(query, {**actions, **observations}, n_samples)
        
        # Change CPT for each root state node
        # for nid in query:
            # Construct CPT by summing over other state variables