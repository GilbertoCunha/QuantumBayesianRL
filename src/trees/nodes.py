from __future__ import annotations

from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from typing import Union
import networkx as nx
import pandas as pd


class TreeNode:
    def __init__(self, value: int, depth: int):
        self.value: int = value
        self.depth: int = depth
        self.parent: TreeNode = None
        self.children: TreeNode = []
        
    def add_parent(self, parent: TreeNode):
        self.parent = parent
        
    def add_children(self, child: TreeNode):
        self.children.append(child)
        
        
class TreeActionNode(TreeNode):
    pass


class TreeObservationNode(TreeNode):
    pass

Node = Union[TreeActionNode, TreeObservationNode]

class Tree:
    def __init__(self, horizon: int, discount: float, action_space: [int], observation_space: [int], root_observation: int):
        # Initialize arguments
        self.horizon: int = horizon
        self.discount: float = discount
        self.action_space: [int] = action_space
        self.observation_space: [int] = observation_space
        
        # Build the tree
        self.node = self.build_tree(root_observation, 0)
        
    def draw_aux(self, node: Node, depth: int):
        r = [(node.value, depth)]
        for i, child in enumerate(node.children):
            r += self.draw_aux(child, depth+1)
        return r
        
    def __repr__(self):
        r = self.draw_aux(self.node, 0)
        r = sorted(r, key=lambda x: x[1])
        r_ = []
        for v, d in r:
            r_.append(f"Value: {v} | Depth: {d}")
        r = "\n".join(r_)
        return r
        
    def build_tree_aux(self, action, depth):
        # Create node
        r = TreeActionNode(action, depth)
        
        # Create children for each observation
        for observation in self.observation_space:
            r.add_children(self.build_tree(observation, depth))
            
        return r
        
    def build_tree(self, observation, depth):
        # Create observation node
        r = TreeObservationNode(observation, depth)
        
        # Add children if depth allows
        if depth < self.horizon:
            for action in self.action_space:
                r.add_children(self.build_tree_aux(action, depth+1))
                
        return r