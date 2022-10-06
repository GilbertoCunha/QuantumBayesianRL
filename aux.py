from src.rl_algorithms.pomdp_lookahead import build_tree, pomdp_lookahead
from get_ddns import get_tiger_ddn, get_robot_ddn, get_gridworld_ddn
from src.utils import get_avg_reward_and_std, belief_update
from src.networks.qbn import QuantumBayesianNetwork as QBN
from src.networks.bn import BayesianNetwork as BN
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


def get_tree(ddn, horizon):
    action_space = ddn.get_space(ddn.action_type)
    observation_space = ddn.get_space(ddn.observation_type)
    tree = build_tree({}, action_space, observation_space, horizon)
    return tree


def get_sample_ratio_aux(ddn, tree, belief_state, n_samples):
    c_r, q_r = 0, 0
    
    # Iterate all action nodes
    for action_tree in tree.children:
        # Iterate all observation nodes:
        action = action_tree.attributes["action"]
        
        # Add to results
        if len(action_tree.children) > 0:
            observation_nodes = ddn.get_nodes_by_type(ddn.observation_type)
            evidence = {**action, **belief_state}
            probs = ddn.query(observation_nodes, evidence, n_samples)[["Prob"]].values
            q_r += (np.sqrt(1 / (probs + 1e-6))).sum()
            c_r += (1 / (probs + 1e-6)).sum()
        
        # Recursive call
        for observation_tree in action_tree.children:
            observation = observation_tree.attributes["observation"]
            new_belief = belief_update(ddn, belief_state, action, observation, n_samples)
            new_cr, new_qr = get_sample_ratio_aux(ddn, observation_tree, new_belief, n_samples)
            c_r += new_cr
            q_r += new_qr
            
    return c_r, q_r


def get_sample_ratio(ddn, tree, belief_state, n_samples):
    c_r, q_r = get_sample_ratio_aux(ddn, tree, belief_state, n_samples)
    if q_r == 0:
        r = 1
    else:
        r = c_r / q_r
    return r


def get_metrics_per_run(ddn, tree, n_samples, reward_samples, time, quantum=False):
    # Calculate metrics for the time-steps
    rs, stds, samples = [], [], []

    # Initialize loop
    description = "Quantum timestep" if quantum else "Classical timestep"
    true_belief = ddn.get_belief_state()
    tbar = tqdm(range(time), total=time, desc=description, position=2, leave=False)
    for _ in tbar:
        # If run is quantum, change number of samples
        ratio = get_sample_ratio(ddn, tree, ddn.get_belief_state(), reward_samples)
        n_samples_ = int(np.ceil(ratio * n_samples))

        # Calculate results
        actions = pomdp_lookahead(ddn, tree, n_samples_)
        avg, cur_std = get_avg_reward_and_std(ddn, ("R", 1), {**actions, **true_belief}, reward_samples)
        
        # Belief update
        observations = ddn.sample_observation(actions)
        ddn.belief_update(actions, observations, n_samples_)
        true_belief = belief_update(ddn, true_belief, actions, observations, reward_samples)

        # Append results
        rs.append(avg)
        stds.append(cur_std)
        samples.append(n_samples_)
        
    rs, stds, samples = np.array(rs), np.array(stds), np.array(samples)
    
    return rs, stds, samples


def get_metrics(ddn, tree, config, num_runs, time):
    # Calculate metrics per run
    r = []
    
    # Get config parameters
    problem_name = config["problem_name"]
    horizon = config["horizon"]
    classical_samples = config["c_samples"]
    reward_samples = config["r_samples"]
    
    # Iterate all runs
    run_bar = tqdm(range(num_runs), total=num_runs, desc=f"{problem_name} runs", position=1, leave=False)
    run_bar.set_postfix(H=horizon, base_samples=classical_samples)
    for run_num in run_bar:
        # Get metrics for specific run
        rs, stds, samples = get_metrics_per_run(ddn, tree, classical_samples, reward_samples, time)
        q_rs, q_stds, q_samples = get_metrics_per_run(ddn, tree, classical_samples, reward_samples, time, True)
        
        # Append to resulting list of dicts
        runs = np.repeat(run_num, time)
        ts = np.arange(time)
        run_dict = [{
            "run": run, 
            "t": t, 
            "r": r, 
            "std": std, 
            "q_r": q_r, 
            "q_std": q_std, 
            "sample": sample, 
            "q_sample": q_sample
        } for (run, t, r, std, q_r, q_std, sample, q_sample) in zip(runs, ts, rs, stds, q_rs, q_stds, samples, q_samples)]
        r += run_dict
    
    return r


def run_config(config, num_runs, time):
    # Extract data from config
    name = config["experiment"]
    discount = config["discount"]
    horizon = config["horizon"]
    
    # Get the ddn
    if name == "tiger":
        ddn = get_tiger_ddn(BN, discount)
        # qddn = get_tiger_ddn(QBN, discount)
    elif name == "robot":
        ddn = get_robot_ddn(BN, discount)
        # qddn = get_robot_ddn(QBN, discount)
    elif name == "gridworld":
        ddn = get_gridworld_ddn(BN, discount)
        # qddn = get_gridworld_ddn(QBN, discount)
    
    # Build the lookahead tree
    tree = get_tree(ddn, horizon)
    
    # Get metrics
    run_dict = get_metrics(ddn, tree, config, num_runs, time)
    
    # Transform configs into dictionary for dataframe
    df = pd.DataFrame([{**config, **run_d} for run_d in run_dict])
        
    # Append results to possibly existing dataframe
    if os.path.isfile("data.csv"):
        df.to_csv("data.csv", mode='a', index=False, header=False)
    else:
        df.to_csv("data.csv", index=False)