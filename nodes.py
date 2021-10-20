from __future__ import annotations
import pandas as pd
import numpy as np

# Defining types
Id = (str, int)

class StateNode:
    def __init__(self, name: str, time: int):
        self.name: str = name
        self.time: int = time
        
    def get_id(self) -> Id:
        return (self.name, self.time)
            
    def get_name(self) -> str:
        return self.name
            
    def get_time(self) -> int:
        return self.time

    def add_pt(self, pt: dict[Union[Id, str], list[int]]):
        self.pt = pd.DataFrame(pt)
        
    def get_sample(self, sample: dict[Id, int]) -> pd.DataFrame:
        # Get the row relative to the current sample where current node is false
        sample = {k: v for k, v in sample.items() if (k in self.pt) and (k != self.get_id())}
        df = self.pt
        for node_id in sample:
            df = df.loc[df[node_id] == sample[node_id]]
        df = df.loc[df[self.get_id()] == 0]
        
        # Generate random number
        number = np.random.uniform()
        r = int(np.random.uniform() > df["Prob"])
        
        return r
        
    
class EvidenceNode(StateNode):
    pass


class ActionNode(StateNode):
    def __init__(self, name: str, time: int, actions: list[int]):
        super().__init__(name, time)
        self.actions = actions
        
    def get_actions(self) -> list[int]:
        return self.actions
        
    def add_value(self, value: int):
        self.pt = pd.DataFrame({self.get_id(): [value, 1-value], "Prob": [1, 0]})


class UtilityNode(StateNode):
    pass