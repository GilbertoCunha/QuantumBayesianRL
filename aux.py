from get_ddns import get_tiger_ddn, get_robot_ddn, get_gridworld_ddn
from src.rl_algorithms.pomdp_lookahead import build_tree, pomdp_lookahead
from src.networks.qbn import QuantumBayesianNetwork as QBN
from src.networks.bn import BayesianNetwork as BN
from src.utils import get_avg_reward_and_std
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np


def get_tree(ddn, horizon):
    horizon = 2
    action_space = ddn.get_space(ddn.action_type)
    observation_space = ddn.get_space(ddn.observation_type)
    tree = build_tree({}, action_space, observation_space, horizon)
    return tree


def get_metrics_per_run(ddn, tree, n_samples, reward_samples, time):
    # Calculate metrics for the time-steps
    avg_r, std = [], []

    tbar = tqdm(range(time), total=time, desc="Timestep", position=2, leave=False)
    for _ in tbar:

        # Classical results
        actions = pomdp_lookahead(ddn, tree, n_samples)
        avg, cur_std = get_avg_reward_and_std(ddn, ("R", 1), {**actions, **ddn.get_belief_state()}, reward_samples)
        observations = ddn.sample_observation(actions)
        ddn.belief_update(actions, observations, n_samples)

        # Append results
        avg_r.append(avg)
        std.append(cur_std)
        
        # Place results in the bar
        tbar.set_postfix(avg_r=avg, std=cur_std)
        
    avg_r, std = np.array(avg_r), np.array(std)
    
    return avg_r, std


def get_metrics(ddn, tree, n_samples, reward_samples, time, num_runs, problem_name, horizon, ratio):
    # Calculate metrics per run
    avg_rs, stds = [], []
    
    # Iterate all runs
    run_bar = tqdm(range(num_runs), total=num_runs, desc=f"{problem_name} runs", position=1, leave=False)
    run_bar.set_postfix(H=horizon, ratio=ratio)
    for run in run_bar:
        # Get metrics for specific run
        avg_r, std = get_metrics_per_run(ddn, tree, n_samples, reward_samples, time)
        
        # Append metrics to list
        avg_rs.append(avg_r)
        stds.append(std)
        
    # Turn lists to arrays
    avg_rs, stds = np.array(avg_rs), np.array(stds)
        
    # Get mean metrics
    avg_rs = np.mean(avg_rs, axis=0)
    stds = np.sqrt(np.mean(std**2, axis=0))
    
    return avg_rs, stds


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


def run_config(config):
    # Extract data from config
    name = config["problem_name"]
    discount = config["discount"]
    horizon = config["horizon"]
    classical_samples = config["classical_samples"]
    ratio = config["ratio"]
    reward_samples = config["reward_samples"]
    time = config["time"]
    num_runs = config["num_runs"]
    
    # Get the ddn
    if name == "Tiger":
        ddn = get_tiger_ddn(BN, discount)
        # qddn = get_tiger_ddn(QBN, discount)
    elif name == "Robot":
        ddn = get_robot_ddn(BN, discount)
        # qddn = get_robot_ddn(QBN, discount)
    elif name == "Gridworld":
        ddn = get_gridworld_ddn(BN, discount)
        # qddn = get_gridworld_ddn(QBN, discount)
    
    # Build the lookahead tree
    tree = get_tree(ddn, horizon)
    
    # Get metrics
    quantum_samples = int(np.ceil(classical_samples**ratio))
    c_avg_r, c_std = get_metrics(ddn, tree, classical_samples, reward_samples, time, num_runs, "Classic " + name, horizon, ratio)
    q_avg_r, q_std = get_metrics(ddn, tree, quantum_samples, reward_samples, time, num_runs, "Quantum " + name, horizon, ratio)
    
    # Save plots for this run
    run_df = pd.DataFrame({
        "c_avg_r": c_avg_r,
        "c_avg_std": c_std,
        "q_avg_r": q_avg_r,
        "q_avg_std": q_std
    })
    # saveplots(c_avg_r, c_std, q_avg_r, q_std, name, horizon, discount, classical_samples, ratio, quantum_samples, num_runs)
    return config, run_df