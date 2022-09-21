from concurrent.futures import ProcessPoolExecutor
from src.utils import product_dict
from aux import run_config
from tqdm import tqdm
import numpy as np

configs = {
    "problem_name": ["Robot", "Gridworld"],
    "discount": [0.8],
    "horizon": [1, 2],
    "classical_samples": [10],
    "ratio": [1.4, 1.6, 1.8, 2],
    "reward_samples": [1000],
    "time": [40],
    "num_runs": [50]
}

# Create list of dictionaries as product of dictionary of lists
total_configs = np.prod([len(v) for _, v in configs.items()])
configs = product_dict(configs)

if __name__ == "__main__":
    # Parallelize code
    with ProcessPoolExecutor() as executor:
        # Iterate each config
        results = list(tqdm(executor.map(run_config, configs), total=total_configs, desc="Iterating configs", position=0, leave=False))