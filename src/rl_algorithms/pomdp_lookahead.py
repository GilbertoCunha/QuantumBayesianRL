from __future__ import annotations
from src.utils import product_dict, belief_update, df_dict_filter, get_expected_reward
from src.trees.tree import Tree
from typing import Union
import pandas as pd

# Types
Id = tuple[str, int]
Value = Union[int, float]
SpaceElement = dict[Id, Value]
Space = dict[Id, list[Value]]
BeliefState = dict[Id, pd.DataFrame]
DDN = "DynamicDecisionNetwork"


def build_tree_aux(action: SpaceElement, action_space: Space, observation_space: Space, horizon: int) -> Tree:
    # Initialize tree
    r = Tree({"type": "action", "action": action})
    
    # Create subtrees
    if horizon > 0:
        for observation in product_dict(observation_space):
            r.add_child(build_tree(observation, action_space, observation_space, horizon))
            
    return r
        

def build_tree(observation: SpaceElement, action_space: Space, observation_space: Space, horizon: int) -> Tree:
    # FIXME: be careful with the time in the node ids for both the observations and actions
    # Initialize tree
    r = Tree({"type": "observation", "observation": observation})
    
    # Create subtrees
    for action in product_dict(action_space):
        r.add_child(build_tree_aux(action, action_space, observation_space, horizon-1))
            
    return r


def q_value(ddn: DDN, tree: Tree, belief_state: BeliefState, n_samples: int) -> Value:
    # TODO: Make sure tree is an action node
    # Create evidence and perform query
    action = tree.get_attribute("action")
    reward_node = ddn.get_nodes_by_type(ddn.reward_type)[0] # TODO: Make sure only one reward node exists
    evidence = {**belief_state, **action}
    
    # Increase value by expected reward
    r = get_expected_reward(ddn, reward_node, evidence, n_samples)
    
    # Increase value due to children nodes
    if len(tree.get_children()) > 0:
        # Calculate observation distribution
        observation_nodes = ddn.get_nodes_by_type(ddn.observation_type)
        observation_df = ddn.query(observation_nodes, evidence, n_samples)
        
        # Iterate every children observation node
        for child in tree.get_children():
            # Calculate probability of observation
            observation = child.get_attribute("observation")
            prob = df_dict_filter(observation_df, observation)
            prob = float(prob["Prob"]) if len(prob) > 0 else 0.0
            
            # Recursive q-value calculation
            new_belief = belief_update(ddn, belief_state, action, observation, n_samples)
            value = max([q_value(ddn, c, new_belief, n_samples) for c in child.get_children()])
            
            # Increase q-value
            r += ddn.get_discount() * prob * value
        
    return r


def pomdp_lookahead(ddn: DDN, tree: Tree, n_samples: int) -> dict[Id, Value]:
    r = None
    best_q = float("-inf")
    
    # Iterate every action node in tree
    for child in tree.get_children():
        action = child.get_attribute("action")
        q = q_value(ddn, child, ddn.get_belief_state(), n_samples)
        
        # Replace best action
        if q > best_q:
            best_q = q
            r = action
            
    return r