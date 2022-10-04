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


def get_metrics_per_run(ddn, qddn, tree, n_samples, reward_samples, time):
    # Calculate metrics for the time-steps
    avg_r, std = [], []
    q_avg_r, q_std = [], []
    ratios, caps = [], []

    # Initialize loop
    true_c_belief, true_q_belief = ddn.get_belief_state(), qddn.get_belief_state()
    tbar = tqdm(range(time), total=time, desc="Timestep", position=2, leave=False)
    for _ in tbar:
        # If run is quantum, change number of samples
        ratio = get_sample_ratio(ddn, tree, ddn.get_belief_state(), reward_samples)
        q_samples = int(np.ceil(ratio * n_samples))
        
        # Cap the maximum number of samples
        if q_samples > n_samples**2:
            q_samples = n_samples**2
            cap = 1
        else:
            cap = 0

        # Calculate results
        actions = pomdp_lookahead(ddn, tree, n_samples)
        q_actions = pomdp_lookahead(qddn, tree, q_samples)
        avg, cur_std = get_avg_reward_and_std(ddn, ("R", 1), {**actions, **true_c_belief}, reward_samples)
        q_avg, q_cur_std = get_avg_reward_and_std(qddn, ("R", 1), {**q_actions, **true_q_belief}, reward_samples)
        
        # Belief update
        observations = ddn.sample_observation(actions)
        q_observations = qddn.sample_observation(q_actions)
        ddn.belief_update(actions, observations, n_samples)
        qddn.belief_update(q_actions, q_observations, n_samples)
        true_c_belief = belief_update(ddn, true_c_belief, actions, observations, reward_samples)
        true_q_belief = belief_update(ddn, true_q_belief, q_actions, q_observations, reward_samples)

        # Append results
        avg_r.append(avg)
        std.append(cur_std)
        q_avg_r.append(q_avg)
        q_std.append(q_cur_std)
        ratios.append(q_samples / n_samples)
        caps.append(cap)
        
    avg_r, std = np.array(avg_r), np.array(std)
    q_avg_r, q_std = np.array(q_avg_r), np.array(q_std)
    ratios, caps = np.array(ratios), np.array(caps)
    
    return avg_r, std, q_avg_r, q_std, ratios, caps


def get_metrics(ddn, qddn, tree, config, num_runs, time, reward_samples):
    # Calculate metrics per run
    avg_rs, stds, q_avg_rs, q_stds, ratios, caps = [], [], [], [], [], []
    
    # Get config parameters
    problem_name = config["problem_name"]
    horizon = config["horizon"]
    classical_samples = config["classical_samples"]
    
    # Iterate all runs
    run_bar = tqdm(range(num_runs), total=num_runs, desc=f"{problem_name} runs", position=1, leave=False)
    run_bar.set_postfix(H=horizon, base_samples=classical_samples)
    for _ in run_bar:
        # Get metrics for specific run
        avg_r, std, q_avg_r, q_std, ratio, cap = get_metrics_per_run(ddn, qddn, tree, classical_samples, reward_samples, time)
        
        # Append metrics to list
        avg_rs.append(avg_r)
        stds.append(std)
        q_avg_rs.append(q_avg_r)
        q_stds.append(q_std)
        ratios.append(ratio)
        caps.append(cap)
        
    # Turn lists to arrays
    avg_r, stds = np.array(avg_rs), np.array(stds)
    q_avg_r, q_stds = np.array(q_avg_rs), np.array(q_stds)
    ratios, caps = np.array(ratios), np.array(caps)
    
    return avg_r, stds, q_avg_r, q_stds, ratios, caps


def run_config(config, num_runs, time, reward_samples):
    # Extract data from config
    name = config["problem_name"]
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
    c_r, c_std, q_r, q_std, ratio, caps = get_metrics(ddn, ddn, tree, config, num_runs, time, reward_samples)
    
    # Save plots for this config
    run_dict = [{
        "run_num": i,
        "time_step": j,
        "c_r": c_r[i,j],
        "c_std": c_std[i,j],
        "q_r": q_r[i,j],
        "q_std": q_std[i,j],
        "ratio": ratio[i,j],
        "cap": caps[i,j]
    } for i in range(num_runs) for j in range(time)]
    
    # Transform configs into dictionary for dataframe
    df = pd.DataFrame([{**config, **run_d} for run_d in run_dict])
        
    # Append results to possibly existing dataframe
    if os.path.isfile("data.csv"):
        df.to_csv("data.csv", mode='a', index=False, header=False)
    else:
        df.to_csv("data.csv", index=False)