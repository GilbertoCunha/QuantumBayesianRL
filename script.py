from concurrent.futures import ProcessPoolExecutor
from src.utils import product_dict
from aux import run_config
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

configs = {
    "problem_name": ["tiger", "robot"],
    "discount": [0.9],
    "horizon": [1, 2, 3],
    "classical_samples": [10, 30, 50],
    "reward_samples": [250],
    "time": [40],
    "num_runs": [50]
}

# Create list of dictionaries as product of dictionary of lists
total_configs = np.prod([len(v) for _, v in configs.items()])
configs = product_dict(configs)

if __name__ == "__main__":
    # Extract results from multiple runs in parallel
    with ProcessPoolExecutor() as executor:
        # Iterate each config
        results = list(tqdm(executor.map(run_config, configs), total=total_configs, desc="Iterating configs", position=0, leave=False))