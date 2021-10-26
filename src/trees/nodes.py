from __future__ import annotations
from typing import Union
import pandas as pd

# Defining types
Id = (str, int)
Node = Union["TreeEvidenceNode", "TreeBeliefNode"]

class TreeEvidenceNode:
    depth: int
    evidence: dict[Id, int]
    prob: float
    utility: float
    children: list[Node]

    def __init__(self, evidence: dict[Id, int], prob: float, depth: int):
        self.depth = depth
        self.evidence = evidence
        self.prob = prob
        self.children = []


class TreeBeliefNode:
    depth: int
    belief: pd.DataFrame
    action: dict[Id, int]
    utility: float
    children: list[Node]

    def __init__(self, belief: dict[Union[Id, str], list[int]], action: dict[Id, int], depth: int):
        self.depth = depth
        self.belief = belief
        self.action = action
        self.children = []
