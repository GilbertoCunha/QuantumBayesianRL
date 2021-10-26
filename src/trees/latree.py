from __future__ import annotations
from src.networks.nodes import ActionNode, EvidenceNode, StateNode
from src.trees.nodes import TreeEvidenceNode, TreeBeliefNode
from src.networks.ddn import DynamicDecisionNetwork
from typing import Type, Union
from itertools import product
import pandas as pd
import numpy as np

# Defining types
Id = (str, int)
NetworkNode = Union[ActionNode, EvidenceNode, StateNode]
TreeNode = Union[TreeEvidenceNode, TreeBeliefNode]


class LookAheadTree:
    root: TreeEvidenceNode
    action_space: pd.DataFrame 
    evidence_space: pd.DataFrame 
    state_space: pd.DataFrame 

    def __init__(self, ddn: DynamicDecisionNetwork, evidence: dict[Id, int]):
        # Define the spaces of actions, evidences and states
        self.action_space = self.get_node_space(ddn, ActionNode, ddn.get_time())
        self.evidence_space = self.get_node_space(ddn, EvidenceNode, ddn.get_time()+1)
        self.state_space = self.get_node_space(ddn, StateNode, ddn.get_time()+1)
        
        # Build the tree
        self.root = TreeEvidenceNode(evidence, 1)

    def get_node_space(self, ddn: DynamicDecisionNetwork, node_type: Type(NetworkNode), time: int) -> pd.DataFrame: 
        nodes = ddn.get_nodes_by_type_and_time(node_type, time)
        value_spaces = [ddn.node_map[node].get_value_space() for node in nodes]
        node_space = np.array(list(product(*value_spaces))).T.tolist()
        node_space = {nodes[i]: node_space[i] for i in range(len(nodes))}
        return pd.DataFrame(node_space)

    def build_tree(self, ddn: DynamicDecisionNetwork, node: TreeNode):
        # if type(node) is TreeEvidenceNode:
           # TODO
        pass

