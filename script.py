from concurrent.futures import ProcessPoolExecutor
from metric_collector import run_config
from src.utils import product_dict
from tqdm import tqdm

# Create configurations for the experiments
num_runs = 40
time = 50
"""configs = {
    "experiment": ["tiger", "robot", "gridworld"],
    "discount": [0.9],
    "horizon": [1, 2, 3],
    "c_samples": [5, 15, 50, 100],
    "r_samples": [250]
}"""
configs = {
    "experiment": ["robot", "gridworld"],
    "discount": [0.9],
    "horizon": [2],
    "c_samples": [50, 100],
    "r_samples": [250]
}

# Create list of dictionaries as product of dictionary of lists
configs = list(product_dict(configs))

configs_ = {
    "experiment": ["gridworld"],
    "discount": [0.9],
    "horizon": [2],
    "c_samples": [5, 15],
    "r_samples": [250]
}

# Create list of dictionaries as product of dictionary of lists
configs += list(product_dict(configs_))

configs_ = {
    "experiment": ["tiger"],
    "discount": [0.9],
    "horizon": [3],
    "c_samples": [5, 15, 50, 100],
    "r_samples": [250]
}

# Create list of dictionaries as product of dictionary of lists
configs += list(product_dict(configs_))

# Create iterator function
def foo(config):
    return run_config(config, num_runs, time)

if __name__ == "__main__":
    # Extract results from multiple runs in parallel
    with ProcessPoolExecutor() as executor:
        # Iterate each config
        _ = list(tqdm(executor.map(foo, configs), total=len(configs), desc="Iterating configs", position=0, leave=False))
