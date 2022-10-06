from concurrent.futures import ProcessPoolExecutor
from src.utils import product_dict
from aux import run_config
from tqdm import tqdm

# Create configurations for the experiments
configs = {
    "experiment": ["tiger", "robot"],
    "discount": [0.9],
    "horizon": [1, 2],
    "c_samples": [5, 15, 50],
    "r_samples": [250]
}
num_runs = 50
time = 40

# Create list of dictionaries as product of dictionary of lists
configs = list(product_dict(configs))
# Create list of dictionaries as product of dictionary of lists
configs_ = {
    "experiment": ["gridworld"],
    "discount": [0.9],
    "horizon": [1, 2],
    "c_samples": [5, 15, 50],
    "r_samples": [250]
}
configs += list(product_dict(configs_))

# Create iterator function
def foo(config):
    return run_config(config, num_runs, time)

if __name__ == "__main__":
    # Extract results from multiple runs in parallel
    with ProcessPoolExecutor() as executor:
        # Iterate each config
        _ = list(tqdm(executor.map(foo, configs), total=len(configs), desc="Iterating configs", position=0, leave=False))