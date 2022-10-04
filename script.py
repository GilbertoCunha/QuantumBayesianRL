from concurrent.futures import ProcessPoolExecutor
from src.utils import product_dict
from aux import run_config
from tqdm import tqdm
import numpy as np

# Create configurations for the experiments
configs = {
    "problem_name": ["tiger", "robot", "gridworld"],
    "discount": [0.9],
    "horizon": [2, 3],
    "classical_samples": [5, 15, 50]
}
reward_samples = 1000
num_runs = 30
time = 30

# Create list of dictionaries as product of dictionary of lists
total_configs = np.prod([len(v) for _, v in configs.items()])
configs = product_dict(configs)

# Create iterator function
def foo(config):
    return run_config(config, num_runs, time, reward_samples)

if __name__ == "__main__":
    # Extract results from multiple runs in parallel
    with ProcessPoolExecutor() as executor:
        # Iterate each config
        _ = list(tqdm(executor.map(foo, configs), total=total_configs, desc="Iterating configs", position=0, leave=False))