from __future__ import annotations
from src.networks.dn import DecisionNetwork

# Define types
Id = str
Value = int


class POMDP(DecisionNetwork):
    
    def __init__(self, discount: float, time: int):
        super().__init__()
        self.discount: float = discount
        self.time: int = time
        
    def get_discount(self) -> float:
        return self.discount
    
    def get_time(self) -> int:
        return self.time
        
    def query_decision(self, *args, **kwargs):
        raise NotImplemented("Method query_decision not implemented for POMDP.")
        
    def get_observation_sample(self, evidence: dict[Id, Value]) -> dict[Id, Value]:
        # Evidence dict is a dictionary of the action values to take
        query = self.get_nodes_by_type(self.observation_type)
        sample = self.query(query, evidence, n_samples=1).to_dict(orient="list")
        sample = {k: v.pop() for k, v in sample.items()}
        return sample
        