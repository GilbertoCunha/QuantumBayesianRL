from __future__ import annotations
from src.networks.ddn import DynamicDecisionNetwork as DDN
from itertools import product
import pandas as pd

# Types
Id = tuple[str, int]
Value = int | float


def product_dict(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))
        
        
def belief_update(ddn: DDN, actions: dict[Id, Value], observations: dict[Id, Value], n_samples: int = 100) -> dict[Id, pd.DataFrame]:
        # TODO: check if actions and observations dict are correct
        
        # Get root state nodes
        query = ddn.get_root_state_nodes()
        
        # Query the next belief-state
        next_belief = ddn.query(query, {**actions, **observations}, n_samples)
        
        # Change CPT for each root state node
        r = {}
        for nid in query:
            # Get columns to keep in node CPT
            nid_query = list(ddn.get_node(nid).get_pt().columns)
            
            # Construct CPT by summing over other variables
            r[nid] = next_belief.groupby(nid_query).sum().reset_index()
                
        return r