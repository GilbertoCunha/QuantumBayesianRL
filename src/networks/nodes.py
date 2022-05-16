from __future__ import annotations
from typing import Type, Union
import pandas as pd
import numpy as np

# Defining types
Id = (str, int)


class Node:
    """
    A class for a Bayesian Network node of a boolean random variable
    """

    def __init__(self, node_id:Union[str,Id], value_range:(int,int), pt:pd.DataFrame = None):
        self.id: Union[str,Id] = node_id
        self.value_range: (int, int) = value_range
        self.pt = pt

    def get_id(self) -> str:
        return self.id

    def get_pt(self) -> pt.DataFrame:
        return self.pt

    def get_value_range(self) -> (int,int):
        return self.value_range

    def get_value_space(self) -> list[int]:
        start, stop = self.get_value_range()
        return list(range(start, stop+1))

    def add_pt(self, pt:dict[str,list[int]]):
        """
        Adds a probability table to this node
        """
        self.pt = pd.DataFrame(pt)

    def get_sample(self, sample:dict[str,int]) -> int:
        """
        Samples this node via the direct sampling algorithm
        given previous acquired samples (of ancester nodes)
        """

        # Get the row relative to the current sample where current node is false
        sample = {k: v for k, v in sample.items() if (
            k in self.get_pt()) and (k != self.get_id())}
        df = self.get_pt()
        for name in sample:
            df = df.loc[df[name] == sample[name]]

        # Get random value from value range
        cum_prob = 0
        number = np.random.uniform()
        for r in range(len(df)):
            cum_prob += df.iloc[r]["Prob"]
            if number < cum_prob:
                break

        return r


class ActionNode(Node):

    def __init__(self, name: str, value_range: (int, int)):
        super().__init__(name, value_range)

    def set_action(self, value: int):
        values = self.get_value_space()
        probs = [int(value==i) for i in range(len(values))]
        self.pt = pd.DataFrame({self.get_id(): values, "Prob": probs})


class UtilityNode(Node):
    pass


class StateNode(Node):
    pass


class EvidenceNode(Node):
    pass