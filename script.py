from concurrent.futures import ProcessPoolExecutor
from src.utils import product_dict
from aux import run_config
from tqdm import tqdm

# Create configurations for the experiments
configs = {
    "problem_name": ["tiger", "robot"],
    "discount": [0.9],
    "horizon": [1, 2, 3],
    "c_samples": [5, 10, 30]
}
reward_samples = 200
num_runs = 30
time = 30

# Create list of dictionaries as product of dictionary of lists
configs = list(product_dict(configs))

# Create list of dictionaries as product of dictionary of lists
configs_ = {
    "problem_name": ["gridworld"],
    "discount": [0.9],
    "horizon": [2],
    "c_samples": [5, 10, 30]
}
configs += list(product_dict(configs_))

# Create iterator function
def foo(config):
    return run_config(config, num_runs, time, reward_samples)

if __name__ == "__main__":
    # Extract results from multiple runs in parallel
    with ProcessPoolExecutor() as executor:
        # Iterate each config
        _ = list(tqdm(executor.map(foo, configs), total=len(configs), desc="Iterating configs", position=0, leave=False))