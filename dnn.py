from bn import BayesianNetwork, BinaryNode
from dn import BinaryActionNode
import pandas as pd


class EvidenceNode(BinaryNode):
    def __init__(self, name, time):
        super().__init__(name)
        self.time = time


class StateNode(EvidenceNode):
    pass


class UtilityNode(EvidenceNode):
    pass


class ActionNode(BinaryActionNode):
    def __init__(self, name, time, actions):
        super().__init__(name, actions)
        self.time = time


class DynamicDecisionNetwork(BayesianNetwork):
    def __init__(self):
        self.graph = {}
        self.nodes = []
        self.edges = []
        self.knowns = {}
        self.time = 0
