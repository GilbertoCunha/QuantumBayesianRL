from __future__ import annotations
from src.networks.dn import DecisionNetwork

Id = str


class POMDP(DecisionNetwork):
    action_type: str = "action"
    state_type: str = "state"
    observation_type: str = "observation"
    reward_type: str = "reward"
    
    def __init__(self, discount: float):
        super().__init__()
        self.discount: float = discount
        self.time: int = None
        
    def get_discount(self) -> float:
        return self.discount
    
    def get_time(self) -> int:
        return self.time
        
    def calculate_time(self):
        return min([v.get_time() for _, v in self.node_map.items()])
    
    def initialize(self):
        super().initialize()
        self.time = self.calculate_time()
        
    def increase_time(self):
        for _, node in self.node_map.items():
            node.increase_time()
        self.time += 1