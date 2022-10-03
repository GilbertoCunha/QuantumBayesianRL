from concurrent.futures import ProcessPoolExecutor
from src.utils import product_dict
from aux import run_config
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

configs = {
    "problem_name": ["robot"],
    "discount": [0.9],
    "horizon": [1, 2, 3],
    "classical_samples": [5, 10, 15],
    "reward_samples": [200],
    "time": [40],
    "num_runs": [30]
}

# Create list of dictionaries as product of dictionary of lists
total_configs = np.prod([len(v) for _, v in configs.items()])
configs = product_dict(configs)

if __name__ == "__main__":
    # Extract results from multiple runs in parallel
    with ProcessPoolExecutor() as executor:
        # Iterate each config
        results = list(tqdm(executor.map(run_config, configs), total=total_configs, desc="Iterating configs", position=0, leave=False))
        
    # Transform configs into dictionary for dataframe
    config_df = {}
    for config, run_dict in results:
        # Add config to dataframe
        for key, value in config.items():
            if key not in config_df:
                config_df[key] = []
            config_df[key].append(value)
            
        # Add result to dataframe
        for key, value in run_dict.items():
            if key not in config_df:
                config_df[key] = []
            config_df[key].append(value)
    config_df = pd.DataFrame(config_df)
        
    # Append results to possibly existing dataframe
    if os.path.isfile("data.h5"):
        data_df = pd.read_hdf("data.h5")
        data_df = pd.concat([data_df, config_df], ignore_index=True)
    else:
        data_df = config_df
        
    # Add data to hdf file
    data_df.to_hdf("data.h5", key="df")