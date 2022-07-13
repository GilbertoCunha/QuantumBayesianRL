from __future__ import annotations
from itertools import product
import pandas as pd

# Types
Id = tuple[str, int]
Value = int | float
BeliefState = dict[Id, pd.DataFrame]
DDN = "DynamicDecisionNetwork"


def product_dict(my_dict):
    keys = my_dict.keys()
    values = my_dict.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))
        

def df_dict_filter(df: pd.DataFrame, dict_filter: dict):
    return df.loc[(df[list(dict_filter)] == pd.Series(dict_filter)).all(axis=1)]
        
        
def belief_update(ddn: DDN, belief_state: BeliefState, actions: dict[Id, Value], observations: dict[Id, Value], n_samples: int = 100) -> dict[Id, pd.DataFrame]:
        # TODO: check if actions and observations dict are correct
        
        # Get next state nodes
        node_filter = lambda _, n: (n.get_type() == ddn.state_type) and (n.get_time() == ddn.get_time() + 1)
        query = ddn.get_nodes_by_key(node_filter)
        print("Got query nodes.")
        
        # Query the next belief-state
        evidence = {**belief_state, **actions, **observations}
        next_belief = ddn.query(query, evidence, n_samples)
        print("Queried new belief.")
        
        # TODO: Check if there is a better way to do this
        # Reduce column time
        col_replace = {(n, t): (n, t-1) for (n, t) in query}
        next_belief.rename(columns=col_replace, inplace=True)
        print("Replaced column names")
        
        # Change CPT for each root state node
        r = {}
        for nid in query:
            # Get columns to keep in node CPT
            nid = col_replace[nid]
            nid_query = list(ddn.get_node(nid).get_pt().columns)
            
            # Construct CPT by summing over other variables
            r[nid] = next_belief.groupby(nid_query).sum().reset_index()
        print("Finished belief updating.")
                
        return r