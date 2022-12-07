from src.rl_algorithms.pomdp_lookahead import build_tree, pomdp_lookahead
from get_ddns import get_tiger_ddn, get_robot_ddn, get_gridworld_ddn
from src.utils import get_avg_reward_and_std, belief_update
from src.networks.qbn import QuantumBayesianNetwork as QBN
from src.networks.bn import BayesianNetwork as BN
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# Define csv path
CSV_PATH = "test.csv"

def get_tree(ddn, horizon):
    action_space = ddn.get_space(ddn.action_type)
    observation_space = ddn.get_space(ddn.observation_type)
    tree = build_tree({}, action_space, observation_space, horizon)
    return tree


def get_sample_coefficients(ddn, tree, belief_state, n_samples, quantum=False):
    r = 0
    
    # Iterate all action nodes
    for action_tree in tree.children:
        # Iterate all observation nodes:
        action = action_tree.attributes["action"]
        
        # Add to results
        if len(action_tree.children) > 0:
            observation_nodes = ddn.get_nodes_by_type(ddn.observation_type)
            evidence = {**action, **belief_state}
            probs = ddn.query(observation_nodes, evidence, n_samples)[["Prob"]].values
            if quantum:
                r += (np.sqrt(1 / (probs + 1e-6))).sum()
            else:
                r += (1 / (probs + 1e-6)).sum()
        
        # Recursive call
        for observation_tree in action_tree.children:
            observation = observation_tree.attributes["observation"]
            new_belief = belief_update(ddn, belief_state, action, observation, n_samples)
            new_r = get_sample_coefficients(ddn, observation_tree, new_belief, n_samples)
            r += new_r
            
    return r


def get_sample_ratio(cr, qr):
    if qr == 0:
        r = 1
    else:
        r = cr / qr
    return r


def get_metrics_per_run(ddn, tree, n_samples, reward_samples, time, quantum=False):
    # Calculate metrics for the time-steps
    rs, stds, samples = [], [], []
    coeffs = []

    # Initialize loop
    description = "Quantum timestep" if quantum else "Classical timestep"
    true_belief = ddn.get_belief_state()
    tbar = tqdm(range(time), total=time, desc=description, position=2, leave=False)
    for _ in tbar:
        # If run is quantum, change number of samples
        if quantum:
            cl = get_sample_coefficients(ddn, tree, ddn.get_belief_state(), reward_samples)
            coeff = get_sample_coefficients(ddn, tree, ddn.get_belief_state(), reward_samples, True)
            ratio = get_sample_ratio(cl, coeff)
            n_samples_ = int(np.floor(ratio * n_samples))
        else:
            coeff = get_sample_coefficients(ddn, tree, ddn.get_belief_state(), reward_samples)
            n_samples_ = n_samples

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
        coeffs.append(coeff)
        
    rs, stds, samples, coeffs = np.array(rs), np.array(stds), np.array(samples), np.array(coeffs)
    
    return rs, stds, samples, coeffs


def get_metrics(ddn, qddn, tree, config, runs, time):
    # Calculate metrics per run
    r = []
    
    # Get config parameters
    problem_name = config["experiment"]
    horizon = config["horizon"]
    classical_samples = config["c_sample"]
    reward_samples = config["r_sample"]
    
    # Iterate all runs
    run_bar = tqdm(runs, total=len(runs), desc=f"{problem_name} runs", position=1, leave=False)
    run_bar.set_postfix(H=horizon, base_samples=classical_samples)
    for run_num in run_bar:
        # Get metrics for specific run
        rs, stds, _, crs = get_metrics_per_run(ddn, tree, classical_samples, reward_samples, time)
        q_rs, q_stds, q_samples, qrs = get_metrics_per_run(qddn, tree, classical_samples, reward_samples, time, True)
        
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
            "q_sample": q_sample,
            "c_l": c_l,
            "q_l": q_l
        } for (run, t, r, std, q_r, q_std, q_sample, c_l, q_l) in zip(runs, ts, rs, stds, q_rs, q_stds, q_samples, crs, qrs)]
        
        # Create dataframe of runs
        run_df = pd.DataFrame([{**config, **run} for run in run_dict])
        
        # Append results to possibly existing dataframe
        if os.path.isfile(CSV_PATH):
            run_df.to_csv(CSV_PATH, mode='a', index=False, header=False)
        else:
            run_df.to_csv(CSV_PATH, index=False)


def run_config(config, num_runs, time):
    # Extract data from config
    name = config["experiment"]
    discount = config["discount"]
    horizon = config["horizon"]
    
    # Get the ddn
    if name == "tiger":
        ddn = get_tiger_ddn(BN, discount)
        qddn = get_tiger_ddn(QBN, discount)
    elif name == "robot":
        ddn = get_robot_ddn(BN, discount)
        qddn = get_robot_ddn(QBN, discount)
    elif name == "gridworld":
        ddn = get_gridworld_ddn(BN, discount)
        qddn = get_gridworld_ddn(QBN, discount)
    
    # Build the lookahead tree
    tree = get_tree(ddn, horizon)
    
    # Get run list
    runs = range(num_runs)
    if os.path.isfile(CSV_PATH):
        # Get existing runs
        df = pd.read_csv(CSV_PATH)
        df = df[df["experiment"] == name]
        ex_runs = df["run"].unique()
        
        # Calculate remaining runs
        runs = [run for run in runs if run not in ex_runs]
    
    # Get metrics
    get_metrics(ddn, ddn, tree, config, runs, time)
