from __future__ import annotations
from typing import Union
import pandas as pd

# Defining types
Id = (str, int)

class TreeEvidenceNode:
    self.evidence: dict[Id, int]
    self.prob: float
    self.utility: float

    def __init__(self, evidence: dict[Id, int], prob: float):
        self.evidence = evidence
        self.prob = prob


class TreeBeliefNode:
    self.belief: pd.DataFrame
    self.action: dict[Id, int]
    self.utility: float

    def __init__(self, belief: dict[Union[Id, str], list[int]], action: dict[Id, int]):
        self.belief = belief
        self.action = action
