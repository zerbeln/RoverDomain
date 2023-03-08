parameters = {}

# Test Parameters
parameters["starting_srun"] = 0  # Which stat run should testing start on (used for parallel testing)
parameters["stat_runs"] = 1  # Total number of runs to perform
parameters["generations"] = 3000  # Number of generations for CCEA in each stat run
parameters["algorithm"] = "DPP"  # Global, Difference, DPP (D++)
parameters["sample_rate"] = 20  # Spacing for collecting performance data during training (every X generations)
parameters["n_configurations"] = 1  # The number of environmental configurations used for training

# Domain parameters
parameters["x_dim"] = 50.0  # Width of the rover map
parameters["y_dim"] = 50.0  # Height of the rover map
parameters["n_rovers"] = 2  # Number of rover agents on map
parameters["n_poi"] = 2   # Number of POIs on map
parameters["steps"] = 30  # Number of time steps rovers take each episode
parameters["world_setup"] = "All"  # Rover_Only, All
parameters["poi_config_type"] = "Two_POI_LR"  # Random, Two_POI_LR, Twp_POI_TB, Four_Corners, Circle
parameters["rover_config_type"] = "Concentrated"  # Random, Concentrated, Fixed

# Rover Parameters
parameters["sensor_model"] = "summed"  # Should either be "density" or "summed"
parameters["angle_res"] = 360 / 4  # Resolution of sensors (determines number of sectors)
parameters["observation_radius"] = 2.0  # Maximum range at which rovers can observe a POI
parameters["dmax"] = 1.0  # Maximum distance a rover can move in a single time step

# Neural network parameters for rover motor control
parameters["n_inp"] = int(2 * (360/parameters["angle_res"]))
parameters["n_hid"] = 12
parameters["n_out"] = 2

# CCEA parameters
parameters["pop_size"] = 40
parameters["mutation_chance"] = 0.1  # Probability that a mutation will occur
parameters["mutation_rate"] = 0.2  # How much a weight is allowed to change
parameters["epsilon"] = 0.1  # For e-greedy selection in CCEA
parameters["n_elites"] = 1  # How many elites to carry over during elite selection

# Post Training Test Parameters
parameters["vis_running"] = True  # True keeps visualizer from closing until you 'X' out of window
