from EvolutionaryAlgorithms.ccea import CCEA
from NeuralNetworks.neural_network import NeuralNetwork
from RoverDomainCore.reward_functions import calc_difference, calc_dpp
from RoverDomainCore.rover_domain import RoverDomain
import numpy as np
from parameters import parameters as p
from global_functions import create_csv_file, save_best_policies


def sample_best_team(rd, pops, networks):
    """
    Sample the performance of the team comprised of the best individuals discovered so far during the learning process
    """

    # Select network weights
    for rv in rd.rovers:
        policy_id = np.argmax(pops[f'EA{rd.rovers[rv].rover_id}'].fitness)
        weights = pops[f'EA{rd.rovers[rv].rover_id}'].population[f'pol{policy_id}']
        networks[f'NN{rd.rovers[rv].rover_id}'].get_weights(weights)

    g_reward = 0
    for cf_id in range(p["n_configurations"]):
        # Reset rovers to configuration initial conditions
        rd.reset_world(cf_id)
        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        for step_id in range(p["steps"]):
            # Get rover actions from neural network
            rover_actions = []
            for rv in rd.rovers:
                action = networks[f'NN{rd.rovers[rv].rover_id}'].run_rover_nn(rd.rovers[rv].observations)
                rover_actions.append(action)

            step_rewards = rd.step(rover_actions)
            # Calculate rewards at current time step
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]

        for p_reward in poi_rewards:
            g_reward += max(p_reward)

    g_reward /= p["n_configurations"]  # Average across configurations

    return g_reward


def rover_global():
    """
    Train rovers in the classic rover domain using the global reward
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for CCEA populations
    pops = {}
    networks = {}
    for rover_id in range(p["n_rovers"]):
        pops[f'EA{rover_id}'] = CCEA(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])
        networks[f'NN{rover_id}'] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        print("Run: %i" % srun)

        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            for pkey in pops:
                pops[pkey].select_policy_teams()
                pops[pkey].reset_fitness()

            # Test each team from CCEA
            for team_number in range(p["pop_size"]):
                # Select network weights
                for rv in rd.rovers:
                    policy_id = int(pops[f'EA{rd.rovers[rv].rover_id}'].team_selection[team_number])
                    weights = pops[f'EA{rd.rovers[rv].rover_id}'].population[f'pol{policy_id}']
                    networks[f'NN{rd.rovers[rv].rover_id}'].get_weights(weights)

                for cf_id in range(p["n_configurations"]):
                    # Reset environment to configuration initial conditions
                    rd.reset_world(cf_id)
                    poi_rewards = np.zeros((p["n_poi"], p["steps"]))
                    for step_id in range(p["steps"]):
                        # Get rover actions from neural network
                        rover_actions = []
                        for rv in rd.rovers:
                            action = networks[f'NN{rd.rovers[rv].rover_id}'].run_rover_nn(rd.rovers[rv].observations)
                            rover_actions.append(action)

                        step_rewards = rd.step(rover_actions)
                        # Calculate rewards at current time step
                        for poi_id in range(p["n_poi"]):
                            poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                    # Update fitness of policies using reward information
                    g_reward = 0
                    for p_reward in poi_rewards:
                         g_reward += max(p_reward)
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(pops[f'EA{rover_id}'].team_selection[team_number])
                        pops[f'EA{rover_id}'].fitness[policy_id] += g_reward

                # Average reward across number of configurations
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(pops[f'EA{rover_id}'].team_selection[team_number])
                    pops[f'EA{rover_id}'].fitness[policy_id] /= p["n_configurations"]

            # Testing Phase (test best agent team found so far) ------------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(rd, pops, networks))
            # --------------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for rover_id in range(p["n_rovers"]):
                pops[f'EA{rover_id}'].down_select()

        # Record Output Files
        create_csv_file(reward_history, "Output_Data/", "Global_Reward.csv")
        for rover_id in range(p["n_rovers"]):
            best_policy_id = np.argmax(pops[f'EA{rover_id}'].fitness)
            weights = pops[f'EA{rover_id}'].population[f'pol{best_policy_id}']
            save_best_policies(weights, srun, f'RoverWeights{rover_id}', rover_id)

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
        pops[f'EA{rover_id}'] = CCEA(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])
        networks[f'NN{rover_id}'] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        print("Run: %i" % srun)

        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            for pkey in pops:
                pops[pkey].select_policy_teams()
                pops[pkey].reset_fitness()

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(p["pop_size"]):
                # Select network weights
                for rv in rd.rovers:
                    policy_id = int(pops[f'EA{rd.rovers[rv].rover_id}'].team_selection[team_number])
                    weights = pops[f'EA{rd.rovers[rv].rover_id}'].population[f'pol{policy_id}']
                    networks[f'NN{rd.rovers[rv].rover_id}'].get_weights(weights)

                for cf_id in range(p["n_configurations"]):
                    # Reset environment to configuration initial conditions
                    rd.reset_world(cf_id)
                    poi_rewards = np.zeros((p["n_poi"], p["steps"]))
                    for step_id in range(p["steps"]):
                        # Get rover actions from neural network
                        rover_actions = []
                        for rv in rd.rovers:
                            action = networks[f'NN{rd.rovers[rv].rover_id}'].run_rover_nn(rd.rovers[rv].observations)
                            rover_actions.append(action)

                        step_rewards = rd.step(rover_actions)
                        # Calculate rewards at current time step
                        for poi_id in range(p["n_poi"]):
                            poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                    # Update fitness of policies using reward information
                    g_reward = 0
                    for p_reward in poi_rewards:
                        g_reward += max(p_reward)
                    d_rewards = calc_difference(rd.pois, g_reward, rd.rover_poi_distances)
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(pops[f'EA{rover_id}'].team_selection[team_number])
                        pops[f'EA{rover_id}'].fitness[policy_id] += d_rewards[rover_id]

                # Average reward across number of configurations
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(pops[f'EA{rover_id}'].team_selection[team_number])
                    pops[f'EA{rover_id}'].fitness[policy_id] /= p["n_configurations"]

            # Testing Phase (test best agent team found so far) ------------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(rd, pops, networks))
            # --------------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()

        # Record Output Files
        create_csv_file(reward_history, "Output_Data/", "Difference_Reward.csv")
        for rover_id in range(p["n_rovers"]):
            best_policy_id = np.argmax(pops[f'EA{rover_id}'].fitness)
            weights = pops[f'EA{rover_id}'].population[f'pol{best_policy_id}']
            save_best_policies(weights, srun, f'RoverWeights{rover_id}', rover_id)

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
        pops[f'EA{rover_id}'] = CCEA(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])
        networks[f'NN{rover_id}'] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:  # Perform statistical runs
        print("Run: %i" % srun)

        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            for pkey in pops:
                pops[pkey].select_policy_teams()
                pops[pkey].reset_fitness()

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(p["pop_size"]):
                # Select network weights
                for rv in rd.rovers:
                    policy_id = int(pops[f'EA{rd.rovers[rv].rover_id}'].team_selection[team_number])
                    weights = pops[f'EA{rd.rovers[rv].rover_id}'].population[f'pol{policy_id}']
                    networks[f'NN{rd.rovers[rv].rover_id}'].get_weights(weights)

                for cf_id in range(p["n_configurations"]):
                    # Reset environment to configuration initial conditions
                    rd.reset_world(cf_id)
                    poi_rewards = np.zeros((p["n_poi"], p["steps"]))
                    for step_id in range(p["steps"]):
                        # Get rover actions from neural network
                        rover_actions = []
                        for rv in rd.rovers:
                            action = networks[f'NN{rd.rovers[rv].rover_id}'].run_rover_nn(rd.rovers[rv].observations)
                            rover_actions.append(action)

                        step_rewards = rd.step(rover_actions)
                        # Calculate rewards at current time step
                        for poi_id in range(p["n_poi"]):
                            poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                    # Update fitness of policies using reward information
                    g_reward = 0
                    for p_reward in poi_rewards:
                        g_reward += max(p_reward)
                    dpp_rewards = calc_dpp(rd.pois, g_reward, rd.rover_poi_distances)
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(pops[f'EA{rover_id}'].team_selection[team_number])
                        pops[f'EA{rover_id}'].fitness[policy_id] += dpp_rewards[rover_id]

                # Average reward across number of configurations
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(pops[f'EA{rover_id}'].team_selection[team_number])
                    pops[f'EA{rover_id}'].fitness[policy_id] /= p["n_configurations"]

            # Testing Phase (test best agent team found so far) ------------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(rd, pops, networks))
            # --------------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()

        # Record Output Files
        create_csv_file(reward_history, "Output_Data/", "DPP_Reward.csv")
        for rover_id in range(p["n_rovers"]):
            best_policy_id = np.argmax(pops[f'EA{rover_id}'].fitness)
            weights = pops[f'EA{rover_id}'].population[f'pol{best_policy_id}']
            save_best_policies(weights, srun, f'RoverWeights{rover_id}', rover_id)

        srun += 1
