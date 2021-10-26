from __future__ import annotations
from typing import Union
import pandas as pd

# Defining types
Id = (str, int)
Node = Union["TreeEvidenceNode", "TreeBeliefNode"]


class TreeEvidenceNode:
    depth: int
    evidence: dict[Id, int]
    prev_action: dict[Id, int]
    prob: float
    utility: float
    children: list[Node]

    def __init__(self, evidence: dict[Id, int], prev_action: dict[Id, int], prob: float, depth: int):
        self.evidence = evidence
        self.prev_action = prev_action
        self.prob = prob
        self.depth = depth
        self.children = []

    def get_depth(self) -> int:
        return self.depth

    def get_evidence(self) -> dict[Id, int]:
        return self.evidence

    def get_prev_action(self) -> dict[Id, int]:
        return self.prev_action

    def add_child(self, node: Node):
        self.children.append(node)


class TreeBeliefNode:
    depth: int
    belief: pd.DataFrame
    action: dict[Id, int]
    utility: float
    children: list[Node]

    def __init__(self, belief: dict[Union[Id, str], list[int]], action: dict[Id, int], depth: int):
        self.belief = belief
        self.action = action
        self.depth = depth
        self.children = []

    def get_depth(self) -> int:
        return self.depth

    def get_action(self) -> dict[Id, int]:
        return self.action

    def get_belief(self) -> pd.DataFrame:
        return self.belief

    def add_child(self, node: Node):
        self.children.append(node)
