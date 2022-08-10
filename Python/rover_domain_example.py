from ccea import Ccea
from RoverDomain_Core.reward_functions import calc_difference, calc_dpp
from RoverDomain_Core.rover_domain import RoverDomain
from rover_neural_network import NeuralNetwork
import numpy as np
from parameters import parameters as p
from global_functions import create_csv_file, save_best_policies, load_saved_policies, create_pickle_file
from Visualizer.visualizer import run_visualizer


def sample_best_team(rd, pops, networks):
    """
    Sample the performance of the team comprised of the best individuals discovered so far during the learning process
    :param rd: Instance of the rover domain
    :param pops: Dictionary containing CCEA populations
    :param networks: Dictionary containing rover neural networks
    :return: Episodic global reward for team of best individuals
    """
    # Reset world to initial conditions
    rd.reset_world()

    # Rover runs initial scan of environment and selects network weights
    for rv in rd.rovers:
        policy_id = np.argmax(pops["EA{0}".format(rd.rovers[rv].rover_id)].fitness)
        weights = pops["EA{0}".format(rd.rovers[rv].rover_id)].population["pol{0}".format(policy_id)]
        networks["NN{0}".format(rd.rovers[rv].rover_id)].get_weights(weights)

    poi_rewards = np.zeros((p["n_poi"], p["steps"]))
    for tstep in range(p["steps"]):
        rover_actions = []
        for rv in rd.rovers:
            action = networks["NN{0}".format(rd.rovers[rv].rover_id)].run_rover_nn(rd.rovers[rv].observations)
            rover_actions.append(action)

        step_rewards = rd.step(rover_actions)
        for poi_id in range(p["n_poi"]):
            poi_rewards[poi_id, tstep] = step_rewards[poi_id]

    g_reward = 0
    for poi_id in range(p["n_poi"]):
        g_reward += max(poi_rewards[poi_id])

    return g_reward


def rover_global():
    """
    Train rovers in the classic rover domain using the global reward
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for CCEA populations and rover neural networks
    pops = {}
    networks = {}
    for rover_id in range(p["n_rovers"]):
        pops["EA{0}".format(rover_id)] = Ccea(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])
        networks["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            for pkey in pops:
                pops[pkey].select_policy_teams()

            # Test each team from CCEA
            for team_number in range(p["pop_size"]):
                # Reset world to initial conditions and select network weights
                rd.reset_world()
                for rv in rd.rovers:
                    policy_id = int(pops["EA{0}".format(rd.rovers[rv].rover_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rd.rovers[rv].rover_id)].population["pol{0}".format(policy_id)]
                    networks["NN{0}".format(rd.rovers[rv].rover_id)].get_weights(weights)

                poi_rewards = np.zeros((p["n_poi"], p["steps"]))  # Track best POI rewards across all time steps
                for tstep in range(p["steps"]):
                    # Get rover actions from neural network
                    rover_actions = []
                    for rv in rd.rovers:
                        rover_id = rd.rovers[rv].rover_id
                        action = networks["NN{0}".format(rover_id)].run_rover_nn(rd.rovers[rv].observations)
                        rover_actions.append(action)

                    # Environment takes in rover actions and returns global reward and next state
                    step_rewards = rd.step(rover_actions)
                    for poi_id in range(p["n_poi"]):
                        poi_rewards[poi_id, tstep] = step_rewards[poi_id]

                # Update fitness of policies using reward information
                g_reward = 0
                for p_reward in poi_rewards:
                     g_reward += max(p_reward)  # Calculate episodic global reward
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    pops["EA{0}".format(rover_id)].fitness[policy_id] = g_reward

            # Testing Phase (sample best agent team found so far) ----------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(rd, pops, networks))
            # --------------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for rover_id in range(p["n_rovers"]):
                pops["EA{0}".format(rover_id)].down_select()

        # Record Output Files
        create_csv_file(reward_history, "Output_Data/", "Global_Reward.csv")
        for rover_id in range(p["n_rovers"]):
            best_policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(best_policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

        srun += 1


def rover_difference():
    """
    Train rovers in classic rover domain using difference rewards
    """

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for CCEA populations
    pops = {}
    networks = {}
    for rover_id in range(p["n_rovers"]):
        pops["EA{0}".format(rover_id)] = Ccea(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])
        networks["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        # Create new population of policies for each rover
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            for pkey in pops:
                pops[pkey].select_policy_teams()

            # Test each team from CCEA
            for team_number in range(p["pop_size"]):
                # Reset world to initial conditions and select network weights
                rd.reset_world()
                for rv in rd.rovers:
                    policy_id = int(pops["EA{0}".format(rd.rovers[rv].rover_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rd.rovers[rv].rover_id)].population["pol{0}".format(policy_id)]
                    networks["NN{0}".format(rd.rovers[rv].rover_id)].get_weights(weights)

                poi_rewards = np.zeros((p["n_poi"], p["steps"]))  # Track best POI rewards across all time steps
                for tstep in range(p["steps"]):
                    # Get rover actions from neural network
                    rover_actions = []
                    for rv in rd.rovers:
                        rover_id = rd.rovers[rv].rover_id
                        action = networks["NN{0}".format(rover_id)].run_rover_nn(rd.rovers[rv].observations)
                        rover_actions.append(action)

                    # Environment takes in rover actions and returns global reward and next state
                    step_rewards = rd.step(rover_actions)
                    for poi_id in range(p["n_poi"]):
                        poi_rewards[poi_id, tstep] = step_rewards[poi_id]

                # Update fitness of policies using reward information
                g_reward = 0
                for p_reward in poi_rewards:
                    g_reward += max(p_reward)  # Calculate episodic global reward
                d_rewards = calc_difference(rd.pois, g_reward, rd.rover_poi_distances)
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    pops["EA{0}".format(rover_id)].fitness[policy_id] = d_rewards[rover_id]

            # Testing Phase (sample best agent team found so far) ----------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(rd, pops, networks))
            # --------------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()

        # Record Output Files
        create_csv_file(reward_history, "Output_Data/", "Difference_Reward.csv")
        for rover_id in range(p["n_rovers"]):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

        srun += 1


def rover_dpp():
    """
    Train rovers in tightly coupled rover domain using D++
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for CCEA populations
    pops = {}
    networks = {}
    for rover_id in range(p["n_rovers"]):
        pops["EA{0}".format(rover_id)] = Ccea(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])
        networks["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:  # Perform statistical runs

        # Create new population of policies for each rover
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            for pkey in pops:
                pops[pkey].select_policy_teams()

            # Test each team from CCEA
            for team_number in range(p["pop_size"]):
                # Reset world to initial conditions and select network weights
                rd.reset_world()
                for rv in rd.rovers:
                    policy_id = int(pops["EA{0}".format(rd.rovers[rv].rover_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rd.rovers[rv].rover_id)].population["pol{0}".format(policy_id)]
                    networks["NN{0}".format(rd.rovers[rv].rover_id)].get_weights(weights)

                poi_rewards = np.zeros((p["n_poi"], p["steps"]))  # Track best POI rewards across all time steps
                for tstep in range(p["steps"]):
                    # Get rover actions from neural network
                    rover_actions = []
                    for rv in rd.rovers:
                        rover_id = rd.rovers[rv].rover_id
                        action = networks["NN{0}".format(rover_id)].run_rover_nn(rd.rovers[rv].observations)
                        rover_actions.append(action)

                    # Environment takes in rover actions and returns global reward and next state
                    step_rewards = rd.step(rover_actions)
                    for poi_id in range(p["n_poi"]):
                        poi_rewards[poi_id, tstep] = step_rewards[poi_id]

                # Update fitness of policies using reward information
                g_reward = 0
                for p_reward in poi_rewards:
                    g_reward += max(p_reward)  # Calculate episodic global reward
                dpp_rewards = calc_dpp(rd.pois, g_reward, rd.rover_poi_distances)
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    pops["EA{0}".format(rover_id)].fitness[policy_id] = dpp_rewards[rover_id]

            # Testing Phase (sample best agent team found so far) ----------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(rd, pops, networks))
            # --------------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()

        # Record Output Files
        create_csv_file(reward_history, "Output_Data/", "DPP_Reward.csv")
        for rover_id in range(p["n_rovers"]):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

        srun += 1


def test_trained_policies():
    """
    Test rovers trained using Global, Difference, or D++ rewards.
    """
    # World Setup
    rd = RoverDomain()  # Create instance of the rover domain
    rd.load_world()

    # Create dictionary of rover neural network instances
    networks = {}
    for rover_id in range(p["n_rovers"]):
        networks["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Data tracking
    reward_history = []  # Keep track of team performance across stat runs
    average_reward = 0  # Track average reward across runs
    final_rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))  # Track rover trajectories

    # Run tests
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        # Load Trained Rover Networks
        for rv in rd.rovers:
            rover_id = rd.rovers[rv].rover_id
            weights = load_saved_policies('RoverWeights{0}'.format(rover_id), rover_id, srun)
            networks["NN{0}".format(rd.rovers[rv].rover_id)].get_weights(weights)

        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        rd.reset_world()
        for step_id in range(p["steps"]):
            # Get rover actions from neural network
            rover_actions = []
            for rv in rd.rovers:
                # Update rover path tracking for visualizer
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 0] = rd.rovers[rv].loc[0]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 1] = rd.rovers[rv].loc[1]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 2] = rd.rovers[rv].loc[2]

                # Get actions from rover neural networks
                rover_id = rd.rovers[rv].rover_id
                action = networks["NN{0}".format(rover_id)].run_rover_nn(rd.rovers[rv].observations)
                rover_actions.append(action)

            # Environment takes in rover actions and returns next state and global reward
            step_rewards = rd.step(rover_actions)
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]

        # Calculate episodic global reward
        g_reward = 0
        for p_reward in poi_rewards:
            g_reward += max(p_reward)  # Calculate episodic global reward
        reward_history.append(g_reward)
        average_reward += g_reward
        srun += 1

    print(average_reward/p["stat_runs"])
    create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths")
    create_csv_file(reward_history, "Output_Data/", "Final_GlobalRewards.csv")
    if p["run_visualizer"]:
        run_visualizer()
