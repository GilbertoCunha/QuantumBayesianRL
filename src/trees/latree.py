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
    lookahead: int
    root: TreeEvidenceNode
    action_space: pd.DataFrame 
    evidence_space: pd.DataFrame 
    state_space: pd.DataFrame 

    def __init__(self, ddn: DynamicDecisionNetwork, evidence: dict[Id, int], lookahead: int):
        # Save lookahead
        self.lookahead = lookahead

        # Define the spaces of actions, evidences and states
        self.action_space = self.get_node_space(ddn, ActionNode, ddn.get_time())
        self.evidence_space = self.get_node_space(ddn, EvidenceNode, ddn.get_time()+1)
        self.state_space = self.get_node_space(ddn, StateNode, ddn.get_time()+1)
        
        # Build the tree
        self.root = TreeEvidenceNode(evidence, 1)

    def get_lookahead(self) -> int:
        return self.lookahead

    def get_node_space(self, ddn: DynamicDecisionNetwork, node_type: Type(NetworkNode), time: int) -> pd.DataFrame: 
        nodes = ddn.get_nodes_by_type_and_time(node_type, time)
        value_spaces = [ddn.node_map[node].get_value_space() for node in nodes]
        node_space = np.array(list(product(*value_spaces))).T.tolist()
        node_space = {nodes[i]: node_space[i] for i in range(len(nodes))}
        return pd.DataFrame(node_space)

    def build_tree(self, ddn: DynamicDecisionNetwork, node: TreeNode, n_samples: int, quantum: bool):
        depth = self.get_depth()

        # Tree is already built
        if depth >= 2 * self.get_lookahead():
            pass

        # In case the current node is an evidence node
        elif type(node) is TreeEvidenceNode:
            # Create query variables (state variables)
            query = ddn.get_nodes_by_type_and_time(StateNode, ddn.get_time()+1)
            depth = node.get_depth()

            # Create a belief node for every possible action
            for i in range(len(self.action_space)):
                # Get the action to this belief node
                action = self.action_space.iloc[[i]].to_dict(orient="list")
                action = {k: v[0] for k, v in action.items()}

                # Query network for belief (need the previous action)
                evidence = {**node.get_evidence(), **node.get_prev_action(), **action}
                belief = ddn.query(query=query, evidence=evidence, n_samples=n_samples, quantum=quantum)

                # Create new belief node
                belief_node = TreeBeliefNode(belief, action, depth+1)
                self.build_tree(ddn, belief_node, n_samples, quantum)
                node.add_child(belief_node)

        # In case the current node is a belief node
        elif type(node) is TreeBeliefNode:
            # Previous action and node depth for the evidence node
            prev_action = node.get_action()
            depth = node.get_depth()

            # Create query variables (evidence variables and state variables)
            state_nodes = ddn.get_nodes_by_type_and_time(StateNode, ddn.get_time()+1)
            query = state_nodes + ddn.get_nodes_by_type_and_time(EvidenceNode, ddn.get_time()+1)
            evidence_pt = ddn.query(query=query, n_samples=n_samples, quantum=quantum)            

            # Turn JPT into CPT
            evidence_pt["Prob"] = evidence_pt.groupby(state_nodes)["Prob"].transform(lambda x: x / x.sum()) 
            
                    
