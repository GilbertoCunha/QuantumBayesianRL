from __future__ import annotations
from src.networks.bn import BayesianNetwork
from tqdm import tqdm
import itertools

# Defining types
Id = str | tuple[str, int]


class DecisionNetwork(BayesianNetwork):
    """Extends the Bayesian network class to create a Decision network.
    Decision networks typically have action, evidence, and utility nodes 
    (which can be defined using the node_type attribute of the DiscreteNode class).
    """
    action_type: str = "action"
    state_type: str = "state"
    observation_type: str = "observation"
    reward_type: str = "reward"


class StaticDecisionNetwork(DecisionNetwork):
    """
    Augments the Decision network class by allowing the extraction of a near-optimal 
    action via the query_decision method.
    It should be used for static decision making.
    This near-optimal action should maximize (or be close to) the expected utility.
    """
    
    def query_decision(self, query: list[Id], evidence: dict[Id, int] = None, n_samples: int = 1000, verbose: bool = False) -> dict[Id, int]:
        """Selects a near-optimal action using the Bayesian network class's inference methods.

        Args:
            query (list[Id]): the query random variables for the inference. You should choose the utility node of the network.
            evidence (dict[Id, int], optional): values for random variables as evidence for the inference. Defaults to None.
            n_samples (int, optional): number of samples to use in the Bayesian network inference. Defaults to 1000.
            verbose (bool, optional): display progress bar for action space iteration. Defaults to False.

        Returns:
            dict[Id, int]: a dictionary containing the near-optimal values for each action random variable.
        """
        # Get all actions for all the action nodes not in evidence
        action_nodes = self.get_nodes_by_type(self.action_type)
        action_space = {a: self.node_dict[a].get_value_space() for a in action_nodes if a not in evidence}
        
        # Create a list of all possible actions to be taken
        keys, values = zip(*action_space.items())
        action_space = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Iterate each set of actions in action space
        results = []
        iterator = tqdm(action_space, total=len(action_space), desc="Iterating actions", leave=True) if verbose else action_space
        for actions in iterator:
            # Add actions to evidence and perform query
            new_evidence = {**evidence, **actions}
            df = self.query(query=[query], evidence=new_evidence, n_samples=n_samples)
            
            # Get expected utility
            eu = float((df[query] * df["Prob"]).sum())
            results.append((actions, eu))
            
        # Get the result with the maximum expected utility
        r = max(results, key=lambda x: x[1])[0]
        
        return r