from aux import run

config = {
    "problem_name": ["Tiger", "Robot", "Gridworld"],
    "discount": [0.8],
    "horizon": [1, 2],
    "classical_samples": [10],
    "ratio": [1.4, 1.6, 1.8, 2],
    "reward_samples": [1000],
    "time": [40],
    "num_runs": [50]
}
run(config)