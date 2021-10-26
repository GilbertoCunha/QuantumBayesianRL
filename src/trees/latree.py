from __future__ import annotations
from src.networks.nodes import ActionNode, EvidenceNode, StateNode
from src.trees.nodes import TreeEvidenceNode, TreeBeliefNode
from src.networks.ddn import DynamicDecisionNetwork

# Defining types
Id = (str, int)


class LookAheadTree:
    self.root: TreeEvidenceNode
    self.action_space: dict[Id, list[int]]
    self.evidence_space: dict[Id, list[int]]
    self.state_space: dict[Id, list[int]]

    def __init__(self, ddn: DynamicDecisionNetwork, evidence: dict[Id, int]):
        self.root = TreeEvidenceNode(evidence, 1)

