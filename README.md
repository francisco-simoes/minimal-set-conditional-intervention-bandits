# Experiments: Minimal Search Space for Conditional Causal Bandits

*Code for the paper: The Minimal Search Space for Conditional Causal Bandits*

The experiments evaluate the method proposed in the paper for identifying the minimal set of nodes worth testing (the mGISS) when performing conditional interventions in causal graphs.

## Table of Contents
- [Citation](#citation)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Running Experiments](#running-experiments)
- [Key Features](#key-features)
- [Results](#results)

## Citation
If you use this code, please cite:
```
@article{simoes2025minimal,
      title={The Minimal Search Space for Conditional Causal Bandits},
      author={Francisco N. F. Q. Simoes and Itai Feigenbaum and Mehdi Dastani and Thijs van Ommen},
      year={2025},
      eprint={...},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={...},
}
```

## Installation
To set up the environment, run:
```sh
pip install -r requirements.txt
```

## Repository Structure
```
├── experiments
│   ├── data
│   │   └── railway_data.csv
│   ├── Images                                        # Resulting plots
│   │   ├── cumulative_regret_curves_asia.png
│   │   ├── cumulative_regret_curves_child.png
│   │   ├── cumulative_regret_curves_sachs.png
│   │   ├── mGISS_fractions_histograms.png
│   │   ├── mGISS_realworld_fractions.png
│   │   └── railway_graph.png
│   ├── mgiss_for_random_graphs.py                    # Testing impact of C4 algorithm on search space size
│   ├── mgiss_for_real_world_graphs.py                # Testing impact of C4 algorithm on search space size
│   ├── Results
│   │   ├── *.pkl                                     # Recorded run histories
│   │   ├── generate_graph_from_pickled_histories.py
│   │   └── README.md
│   ├── run_cond_int_ucb_on_bnlearn_datasets.py       # Run CondIntUCB on bnlearn datasets
│   └── SlurmScripts                                  # For running in cluster
│       ├── README.md
│       └── slurm_cond_int_ucb_bnlearn.slurm
├── Pipfile                                           # In case you prefer to use pipenv instead of pip
├── Pipfile.lock
├── requirements_dev.txt
└── requirements.txt
```


## Running Experiments
Each of the Python scripts in `experiments/` corresponds to an experiment.
You may want to edit the script constants at the start of the script (written with capital letters).

## Key Features
- **Minimal Search Space Computation**: Implements the C4 algorithm to compute the minimal set of nodes that need to be tested.
- **Evaluation on Real-World and Synthetic Graphs**: Experiments assess the reduction in search space and improvements in regret performance.
- **Conditional Causal Bandits**: Integrates the minimal search space with a UCB-based bandit algorithm for conditional intervention selection.

## Results
The results include:
- Reduction in search space on real-world and synthetic graphs .
- Performance of the conditional causal bandit algorithm using the reduced search space (improved cumulative regret compared to baselines).
