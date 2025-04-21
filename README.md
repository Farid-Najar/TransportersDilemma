# Transporters Dilemma

This repository contains the implementation and experimental setup for the "Demand Selection for VRP with Emission Quota" research project.

## Running Experiments

The main experiments for this project can be run by executing the `experiments.py` script:

```bash
python /TransportersDilemma/experiments.py
```

This script will run a series of experiments with different configurations:

1. **Comparison Experiments**:
   - Various configurations with different K values (20, 50, 100)
   - Different retain rates (1.0, 0.8)
   - All using real data

2. **Simulated Annealing TSP Experiments**:
   - Configurations with K values of 20, 50, and 100
   - Using real data

3. **Learning Strategy Experiments**:
   - Least Reward-Inaction (LRI) algorithm with K values of 20, 50, and 100
   - EXP3 algorithm with K values of 20, 50, and 100
   - All configured for TSP with real data

4. **Reinforcement Learning Experiments** (currently commented out):
   - Various configurations for training RL agents
   - Different observation modes and parameters

## Experiment Parameters

The main parameters used in the experiments include:

- **K**: Number of routes/options (20, 50, or 100)
- **n_simulation**: Number of simulation runs (set to 100)
- **T**: Number of time steps for learning algorithms (10,000 or 15,000)
- **retain**: Retention rate for certain experiments (1.0 or 0.8)
- **real_data**: Flag to use real-world data (set to True)

## Additional Information

The experiments utilize several algorithms implemented in the project:
- Comparison methods from `compare_game_SA`
- Learning strategies (LRI, EXP3) from `GameLearning`
- Simulated Annealing for TSP problems
- Reinforcement Learning approaches (in the commented section)

To modify experiment parameters or enable the RL experiments, edit the `experiments.py` file directly.