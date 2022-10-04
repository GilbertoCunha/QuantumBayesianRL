from src.rl_algorithms.pomdp_lookahead import build_tree, pomdp_lookahead
from get_ddns import get_tiger_ddn, get_robot_ddn, get_gridworld_ddn
from src.utils import get_avg_reward_and_std, belief_update
from src.networks.qbn import QuantumBayesianNetwork as QBN
from src.networks.bn import BayesianNetwork as BN
import matplotlib.pyplot as plt
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
    avg_r, std, sample_nr, caps = [], [], [], []

    tbar = tqdm(range(time), total=time, desc="Timestep", position=2, leave=False)
    for _ in tbar:
        cap = 0
        # If run is quantum, change number of samples
        if quantum:
            ratio = get_sample_ratio(ddn, tree, ddn.get_belief_state(), reward_samples)
            samples = int(np.ceil(ratio * n_samples))
            
            # Cap the maximum number of samples
            if samples > n_samples**2:
                samples = n_samples**2
                cap = 1
        else:
            samples = n_samples

        # Calculate results
        actions = pomdp_lookahead(ddn, tree, samples)
        avg, cur_std = get_avg_reward_and_std(ddn, ("R", 1), {**actions, **ddn.get_belief_state()}, reward_samples)
        observations = ddn.sample_observation(actions)
        ddn.belief_update(actions, observations, samples)

        # Append results
        avg_r.append(avg)
        std.append(cur_std)
        sample_nr.append(samples)
        caps.append(cap)
        
        # Place results in the bar
        tbar.set_postfix(avg_r=avg, std=cur_std)
        
    avg_r, std, sample_nr, caps = np.array(avg_r), np.array(std), np.array(sample_nr), np.array(caps)
    
    return avg_r, std, sample_nr, caps


def get_metrics(ddn, tree, config, num_runs, time, reward_samples, quantum=False):
    # Calculate metrics per run
    avg_rs, stds, sample_nr, caps = [], [], [], []
    
    # Get config parameters
    problem_name = config["problem_name"]
    horizon = config["horizon"]
    classical_samples = config["classical_samples"]
    
    # Iterate all runs
    run_bar = tqdm(range(num_runs), total=num_runs, desc=f"{problem_name} runs", position=1, leave=False)
    run_bar.set_postfix(H=horizon, base_samples=classical_samples)
    for _ in run_bar:
        # Get metrics for specific run
        avg_r, std, samples, cap = get_metrics_per_run(ddn, tree, classical_samples, reward_samples, time, quantum)
        
        # Append metrics to list
        avg_rs.append(avg_r)
        stds.append(std)
        sample_nr.append(samples)
        caps.append(cap)
        
    # Turn lists to arrays
    avg_r, stds, sample_nr, caps = np.array(avg_rs), np.array(stds), np.array(sample_nr), np.array(caps)
    
    return avg_r, stds, sample_nr, caps


def saveplots(c_avg_r, c_std, q_avg_r, q_std, name, horizon, discount, classical_samples, ratio, quantum_samples, num_runs):
    
    # Parameters for the plot
    textstr = f"$H = {horizon}$"
    textstr += f"\n$\gamma={discount}$"
    textstr += f"\nClassical samples={classical_samples}"
    textstr += f"\nQuantum samples={quantum_samples}"
    textstr += f"\nRuns={num_runs}"
    
    # Make figure name
    figname = name + f"H{horizon}"
    figname += f"Ratio{ratio}"
    figname.replace(".","dot")
    
    # Get cummulative rewards and standard deviations
    c_cum_r, q_cum_r = np.cumsum(c_avg_r), np.cumsum(q_avg_r)
    c_cum_std, q_cum_std = np.sqrt(np.cumsum(np.array(c_std)**2)), np.sqrt(np.cumsum(np.array(q_std)**2))

    # Making the plot for cummulative rewards
    x = range(len(c_cum_r))
    plt.title("Cumulative reward over time")
    plt.xlabel("Time-step $t$")
    plt.ylabel("Cumulative reward")
    plt.plot(x, c_cum_r, label="classic")
    plt.fill_between(x, c_cum_r-c_cum_std, c_cum_r+c_cum_std, alpha=0.2)
    plt.plot(x, q_cum_r, label="quantum")
    plt.fill_between(x, q_cum_r-q_cum_std, q_cum_r+q_cum_std, alpha=0.2)
    plt.gcf().text(0.95, 0.7, textstr, fontsize=10)
    plt.legend()
    plt.savefig(f"plots/{name}/{figname}.png", dpi=250, bbox_inches="tight")
    plt.clf()
    
    # Calculate cumulative difference avg and std
    cum_diff_avg, cum_diff_std = q_cum_r - c_cum_r, np.sqrt(np.cumsum((q_std - c_std)**2))

    # Plot the differences
    plt.title("Cumulative reward difference over time")
    plt.xlabel("Time-step $t$")
    plt.ylabel("Cumulative reward difference")
    plt.plot(x, cum_diff_avg)
    plt.fill_between(x, cum_diff_avg-cum_diff_std, cum_diff_avg+cum_diff_std, alpha=0.2)
    plt.gcf().text(0.95, 0.7, textstr, fontsize=10)
    plt.savefig(f"plots/{name}/{figname}Difference.png", dpi=250, bbox_inches="tight")
    plt.clf()


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
    c_r, c_std, c_s, _ = get_metrics(ddn, tree, config, num_runs, time, reward_samples)
    q_r, q_std, q_s, caps = get_metrics(ddn, tree, config, num_runs, time, reward_samples, True)
    ratio = q_s / c_s
    
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