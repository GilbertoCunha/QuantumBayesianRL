from __future__ import annotations
from src.networks.nodes import ActionNode, EvidenceNode, StateNode
from src.trees.nodes import TreeEvidenceNode, TreeBeliefNode
from src.networks.ddn import DynamicDecisionNetwork
from typing import Type, Union
from itertools import product
from functools import reduce
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

    def __init__(self, ddn: DynamicDecisionNetwork, evidence: dict[Id, int], prev_action: dict[Id, int], lookahead: int, n_samples: int):
        # Save lookahead
        self.lookahead = lookahead

        # Define the spaces of actions, evidences and states
        self.action_space = self.get_node_space(ddn, ActionNode, ddn.get_time())
        self.evidence_space = self.get_node_space(ddn, EvidenceNode, ddn.get_time()+1)
        self.state_space = self.get_node_space(ddn, StateNode, ddn.get_time()+1)

        # Build the tree
        self.root = TreeEvidenceNode(evidence, prev_action, 1, 0)
        self.build(ddn, self.root, n_samples)

    def get_lookahead(self) -> int:
        return self.lookahead

    def get_node_space(self, ddn: DynamicDecisionNetwork, node_type: Type(NetworkNode), time: int) -> pd.DataFrame:
        # Returns a DataFrame of every possible combination of values the nodes of a certain type (and of a certain time) can take
        nodes = ddn.get_nodes_by_type_and_time(node_type, time)
        value_spaces = [ddn.node_map[node].get_value_space() for node in nodes]
        node_space = np.array(list(product(*value_spaces))).T.tolist()
        node_space = {nodes[i]: node_space[i] for i in range(len(nodes))}
        return pd.DataFrame(node_space)

    def build(self, ddn: DynamicDecisionNetwork, node: TreeNode, n_samples: int):
        depth = node.get_depth()

        # Tree is already built
        if depth >= 2 * self.get_lookahead() - 1:
            pass

        # In case the current node is an evidence node
        elif type(node) is TreeEvidenceNode:
            # Create query variables (state variables)
            query = ddn.get_nodes_by_type_and_time(StateNode, ddn.get_time()+1)

            # Create a belief node for every possible action
            for i in range(len(self.action_space)):
                # Get the action to this belief node
                action = self.action_space.iloc[[i]].to_dict(orient="list")
                action = {k: v[0] for k, v in action.items()}

                # Query network for belief (need the previous action)
                evidence = {**node.get_evidence(), **node.get_prev_action(), **action}
                belief = ddn.query(query=query, evidence=evidence, n_samples=n_samples)

                # Rename columns
                new_columns = {(c[0], c[1]): (c[0], c[1]-1) for c in belief.columns if c != "Prob"}
                belief = belief.rename(columns=new_columns)

                # Create new belief node
                belief_node = TreeBeliefNode(belief, action, depth+1)
                self.build(ddn, belief_node, n_samples)
                node.add_child(belief_node)

        # In case the current node is a belief node
        elif type(node) is TreeBeliefNode:
            # Previous action and node depth for the evidence node
            prev_action = node.get_action()
            prev_action = {(n,t-1):v for (n,t),v in prev_action.items()}
    
            # Get the ddn only for the current time-step
            cur_ddn = ddn.get_ddn_time_slice(ddn.get_time())

            # Create query variables (evidence variables and state variables)
            state_nodes = cur_ddn.get_nodes_by_type_and_time(StateNode, ddn.get_time())
            evidence_nodes = cur_ddn.get_nodes_by_type_and_time(EvidenceNode, ddn.get_time())
            query = state_nodes + evidence_nodes
            evidence_pt = cur_ddn.query(query=query, evidence=prev_action, n_samples=n_samples)

            # Turn JPT into CPT
            evidence_pt["Prob"] = evidence_pt.groupby(state_nodes)["Prob"].transform(lambda x: x / x.sum())
            
            # Multiply belief state probabilities to CPT
            belief = node.get_belief()
            for i in range(len(belief)):
                # Create query
                df_query = [evidence_pt[c] == belief.iloc[i][c] for c in belief.columns if c != "Prob"]
                df_query = reduce(lambda x,y: x & y, df_query)

                # Change probabilities
                evidence_eval = evidence_pt.eval(df_query)
                evidence_pt.loc[evidence_eval, "Prob"] *= belief.iloc[0]["Prob"]

            # Group by evidence states and normalize
            evidence_pt = evidence_pt.groupby(evidence_nodes)[["Prob"]].agg("sum").reset_index()
            evidence_pt["Prob"] /= evidence_pt["Prob"].sum()

            # Create a new evidence state for each evidence in the probability table
            for i in range(len(evidence_pt)):
                # Get evidence and probability for each node
                evidence = evidence_pt[evidence_nodes].iloc[i].to_dict()
                trans_prob = evidence_pt.iloc[i]["Prob"]

                # Create node
                evidence_node = TreeEvidenceNode(evidence, prev_action, trans_prob, depth+1)
                self.build(ddn, evidence_node, n_samples)
                node.add_child(evidence_node)