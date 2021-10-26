from __future__ import annotations
import pandas as pd
import numpy as np

# Defining types
Id = (str, int)


class StaticNode:
    """
    A class for a Bayesian Network node of a boolean random variable
    """

    def __init__(self, name: str, value_range: (int, int)):
        self.name: str = name
        self.value_range: (int, int) = value_range

    def get_id(self) -> str:
        return self.name

    def get_pt(self) -> pt.DataFrame:
        return self.pt

    def get_value_range(self) -> (int, int):
        return self.value_range

    def get_value_space(self) -> list[int]:
        start, stop = self.get_value_range()
        return list(range(start, stop+1))

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
        sample = {k: v for k, v in sample.items() if (
            k in self.pt) and (k != self.get_id())}
        df = self.pt
        for name in sample:
            df = df.loc[df[name] == sample[name]]
        # df = df.loc[df[self.get_id()] == 0]

        # Generate random number
        # number = np.random.uniform()
        # r = int(np.random.uniform() > df["Prob"])

        # Get random value from value range
        cum_prob = 0
        number = np.random.uniform()
        for r in range(len(df)):
            cum_prob += df.iloc[r]["Prob"]
            if number < cum_prob:
                break

        return r


class StaticActionNode(StaticNode):

    def __init__(self, name: str, value_range: (int, int)):
        super().__init__(name, value_range)

    def add_value(self, value: int):
        values = self.get_value_space() 
        probs = [int(value==i) for i in range(len(values))]
        self.pt = pd.DataFrame({self.name: values, "Prob": probs})


class StaticUtilityNode(StaticNode):
    pass


class StateNode(StaticNode):
    def __init__(self, name: str, time: int, value_range: (int, int)):
        self.name: str = name
        self.time: int = time
        self.value_range: (int, int) = value_range

    def get_id(self) -> Id:
        return (self.name, self.time)

    def get_name(self) -> str:
        return self.name

    def get_time(self) -> int:
        return self.time


class EvidenceNode(StateNode):
    pass


class ActionNode(StateNode):
    def __init__(self, name: str, time: int, value_range: (int, int)):
        super().__init__(name, time, value_range)

    def add_value(self, value: int):
        values = self.get_value_space() 
        probs = [int(value==i) for i in range(len(values))]
        self.pt = pd.DataFrame({self.get_id(): values, "Prob": probs})


class UtilityNode(StateNode):
    pass

