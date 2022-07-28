parameters = {}

# Test Parameters
parameters["starting_srun"] = 0  # Which stat run should testing start on (used for parallel testing)
parameters["stat_runs"] = 1  # Total number of runs to perform
parameters["generations"] = 200  # Number of generations for CCEA in each stat run
parameters["algorithm"] = "Global"  # Global, Difference, DPP (D++)
parameters["sample_rate"] = 20  # Spacing for collecting performance data during training (every X generations)

# Domain parameters
parameters["x_dim"] = 50.0  # X-Dimension of the rover map
parameters["y_dim"] = 50.0  # Y-Dimension of the rover map
parameters["n_rovers"] = 3  # Number of rovers on map
parameters["n_poi"] = 2  # Number of POIs on map
parameters["steps"] = 20  # Number of time steps rovers take each episode
parameters["poi_config_type"] = "Random"  # Random, Two_POI, Four_Corners, Circle, Con_Circle
parameters["rover_config_type"] = "Random"  # Random, Concentrated, Four_Quadrants

# Rover Parameters
parameters["sensor_model"] = "summed"  # Should either be "density" or "summed"
parameters["angle_res"] = 360 / 4  # Resolution of sensors (determines number of sectors)
parameters["observation_radius"] = 4.0  # Maximum range at which rovers can observe a POI
parameters["dmax"] = 1.0  # Maximum distance a rover can move in a single time step

# Neural network parameters for rover motor control
parameters["n_inp"] = int(2 * (360 / parameters["angle_res"]))
parameters["n_hid"] = 10
parameters["n_out"] = 2

# CCEA parameters
parameters["pop_size"] = 40
parameters["mutation_chance"] = 0.1  # Probability that a mutation will occur
parameters["mutation_rate"] = 0.2  # How much a weight is allowed to change
parameters["epsilon"] = 0.1  # For e-greedy selection in CCEA
parameters["n_elites"] = 1  # How many elites to carry over during selection
