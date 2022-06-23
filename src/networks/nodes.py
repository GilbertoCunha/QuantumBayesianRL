from __future__ import annotations
from typing import Union
import pandas as pd
import numpy as np

# Defining types
Id = Union[str, tuple[str, int]]

# FIXME: This code assumes the PT's column name for the probability column is "Prob".
class DiscreteNode:
    """
    A class for a Bayesian Network node of a discrete random variable.
    """

    def __init__(self, node_id: Id, node_type: str, value_space: list[float], pt: pd.DataFrame = None):
        self.id: Id = node_id
        self.type = node_type
        self.value_space: list[float] = value_space
        self.pt: pd.DataFrame = pt

    def get_id(self) -> Id:
        return self.id

    def get_pt(self) -> pd.DataFrame:
        return self.pt
    
    def get_type(self) -> str:
        return self.type

    def get_value_space(self) -> list[float]:
        return self.value_space

    def add_pt(self, pt: dict[Id, list[int]]):
        self.pt = pd.DataFrame(pt)
        
    def fix_value(self, value: int):
        values = self.get_value_space()
        probs = [int(value==i) for i in range(len(values))]
        self.pt = pd.DataFrame({self.get_id(): values, "Prob": probs})

    def get_sample(self, sample: dict[Id, int]) -> int:
        """
        Samples this node via the direct sampling algorithm
        given previous acquired samples (of parent nodes).
        
        TODO:
            -> Check if every parent node is in the input sample.
            -> Guarantee that the filtered df represents a probability distribution.
        """

        # Filter node's pt to match the evidence
        df = self.get_pt()
        sample = {k: v for k, v in sample.items() if (k in df) and (k != self.get_id())}
        for name in sample:
            df = df.loc[df[name] == sample[name]]

        # Generate random [0, 1] real number.
        # Use that real number to determine the sample to extract
        # by accumulating the probabilities of each row of the filtered df.
        r, cum_prob = 0, 0
        number = np.random.uniform()
        for i in range(len(df)):
            cum_prob += df.iloc[i]["Prob"]
            if number < cum_prob:
                r = df.iloc[i][self.get_id()]
                break

        return r