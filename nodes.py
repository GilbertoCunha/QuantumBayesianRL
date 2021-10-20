from __future__ import annotations
import pandas as pd
import numpy as np

# Defining types
Id = (str, int)


class StaticNode:
    """
    A class for a Bayesian Network node of a boolean random variable
    """
    
    def __init__(self, name: str):
        self.name: str = name
        
    def get_id(self) -> str:
        return self.name
        
    def add_pt(self, pt: dict[str, list[int]]):
        """
        Adds a probability table to this node
        """
        self.pt = pd.DataFrame(pt)
    
    def get_sample(self, sample: dict[str, int]) -> int:
        """
        Samples this node via the direct sampling algorithm
        given previous acquired samples (of ancester nodes)
        """
        
        # Get the row relative to the current sample where current node is false
        sample = {name: sample[name] for name in sample if (name in self.pt and name != self.get_id())}
        df = self.pt
        for name in sample:
            df = df.loc[df[name] == sample[name]]
        df = df.loc[df[self.get_id()] == 0]
        
        # Generate random number
        number = np.random.uniform()
        r = int(np.random.uniform() > df["Prob"])
        
        return r
    

class StaticActionNode(StaticNode):
    
    def __init__(self, name: str, actions: list[int]):
        super().__init__(name)
        self.actions = actions
        
    def get_action(self) -> list[int]:
        return self.actions
        
    def add_value(self, value: int):
        self.pt = pd.DataFrame({self.name: [value, 1-value], "Prob": [1, 0]})


class StaticUtilityNode(StaticNode):
    pass


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
        
    def get_sample(self, sample: dict[Id, int]) -> int:
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