from __future__ import annotations
from src.networks.ddn import DynamicDecisionNetwork as DDN
from src.utils import product_dict
from src.trees.tree import Tree
import pandas as pd

# Types
Id = tuple[str, int]
Value = int | float
SpaceElement = dict[Id, Value]
Space = dict[Id, list[Value]]


def build_tree_aux(action: SpaceElement, action_space: Space, observation_space: Space, horizon: int) -> Tree:
    # Initialize tree
    r = Tree(None)
    r.add_attributes({"type": "action", "action": action})
    
    # Create subtrees
    if horizon > 0:
        for observation in product_dict(**observation_space):
            r.add_child(build_tree(observation, action_space, observation_space, horizon))
            
    return r
        

def build_tree(observation: SpaceElement, action_space: Space, observation_space: Space, horizon: int) -> Tree:
    # Initialize tree
    r = Tree(None)
    r.add_attributes({"type": "observation", "observation": observation})
    
    # Create subtrees
    for action in product_dict(**action_space):
        r.add_child(build_tree_aux(action, action_space, observation_space, horizon-1))
            
    return r


def calculate_q_values(ddn: DDN, tree: Tree, belief_state: dict[Id, pd.DataFrame], n_samples: int) -> Value:
    r = 0
    
    # In case of an action node
    if tree.get_attribute("type") == "action":
        # Get reward, action and root state nodes
        root_state_nodes = ddn.get_root_state_nodes()
        action_nodes = ddn.get_nodes_by_type(DDN.action_type)
        reward_node = ddn.get_nodes_by_type(DDN.reward_type)[0] # TODO: Make sure only one reward node
        
        # Create evidence and perform query
        evidence = {root_state_nodes: belief_state, action_nodes: tree.get_attribute("action")}
        reward_df = ddn.query(query=[reward_node], evidence=evidence, n_samples=n_samples)
        
        # Increase value by expected reward
        r += (reward_df[reward_node] * reward_df["Prob"]).sum()
        
        # Increase value due to children nodes
        if len(tree.get_children()) > 0:
            # Calculate observation distribution
            observation_nodes = ddn.get_nodes_by_type(DDN.observation_type)
            observation_df = ddn.query(query=observation_nodes, evidence=evidence, n_samples=n_samples)
            
            # Iterate every children observation node
            for child in tree.get_children():
                prob = 0.2 # TODO: calculate probability properly
                new_belief = ddn.belief_update(tree.get_attribute("action"), child.get_attribute("observation"), n_samples)
                value = calculate_q_values(ddn, child, new_belief, n_samples)
                r += ddn.get_discount() * prob * value
           
    else: # This else assumes that tree.attributes["type"] == "observation"
        r = max([calculate_q_values(ddn, child, belief_state, n_samples) for child in tree.get_children()])
        
    return r