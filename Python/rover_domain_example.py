from ccea import Ccea
from RoverDomain_Core.reward_functions import calc_difference, calc_dpp
from RoverDomain_Core.rover_domain import RoverDomain
import numpy as np
from parameters import parameters as p
from global_functions import create_csv_file, save_best_policies
import time


def sample_best_team(rd, pops):
    """
    Sample the performance of the team comprised of the best individuals discovered so far during the learning process
    :param rd: Instance of the rover domain
    :param pops: CCEA populations
    :return: global reward for team of best individuals
    """
    # Reset rovers to initial conditions
    for rk in rd.rovers:
        rd.rovers[rk].reset_rover()

    # Rover runs initial scan of environment and selects network weights
    for rk in rd.rovers:
        policy_id = np.argmax(pops["EA{0}".format(rd.rovers[rk].self_id)].fitness)
        weights = pops["EA{0}".format(rd.rovers[rk].self_id)].population["pol{0}".format(policy_id)]
        rd.rovers[rk].get_weights(weights)

    poi_rewards = np.zeros((p["n_poi"], p["steps"]))
    for step_id in range(p["steps"]):
        # Rovers scan environment to collect state information
        for rk in rd.rovers:
            rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
        for poi in rd.pois:
            rd.pois[poi].update_observer_distances(rd.rovers)

        # Calculate rewards at current time step
        step_rewards = rd.calc_global()
        for poi_id in range(p["n_poi"]):
            poi_rewards[poi_id, step_id] = step_rewards[poi_id]

        # Each rover acts according to policy
        for rk in rd.rovers:
            rd.rovers[rk].step(rd.world_x, rd.world_y)

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

    # Create dictionary for CCEA populations
    pops = {}
    for rover_id in range(p["n_rovers"]):
        pops["EA{0}".format(rover_id)] = Ccea(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    run_times = []
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        start_time = time.time()

        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            for pkey in pops:
                pops[pkey].select_policy_teams()

            # Test each team from CCEA
            for team_number in range(p["pop_size"]):
                # Reset rovers to initial conditions and select network weights
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                    policy_id = int(pops["EA{0}".format(rd.rovers[rk].self_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rd.rovers[rk].self_id)].population["pol{0}".format(policy_id)]
                    rd.rovers["R{0}".format(rd.rovers[rk].self_id)].get_weights(weights)

                poi_rewards = np.zeros((p["n_poi"], p["steps"]))
                for step_id in range(p["steps"]):
                    # Rovers scan environment to collect state information
                    for rk in rd.rovers:
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)

                    # Calculate rewards at current time step
                    step_rewards = rd.calc_global()
                    for poi_id in range(p["n_poi"]):
                        poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                    # Each rover acts according to policy
                    for rk in rd.rovers:
                        rd.rovers[rk].step(rd.world_x, rd.world_y)

                # Update fitness of policies using reward information
                g_reward = 0
                for poi_id in range(p["n_poi"]):
                     g_reward += max(poi_rewards[poi_id])
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    pops["EA{0}".format(rover_id)].fitness[policy_id] = g_reward

            # Testing Phase (test best agent team found so far) ------------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(rd, pops))
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
        end_time = time.time()
        run_times.append(end_time - start_time)

    create_csv_file(run_times, "Output_Data/", "GlobalRunTimes.csv")


def rover_difference():
    """
    Train rovers in classic rover domain using difference rewards
    """

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for CCEA populations
    pops = {}
    for rover_id in range(p["n_rovers"]):
        pops["EA{0}".format(rover_id)] = Ccea(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    run_times = []
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        start_time = time.time()

        # Create new population of policies for each rover
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            for pkey in pops:
                pops[pkey].select_policy_teams()

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(p["pop_size"]):
                # Reset rovers to initial conditions and select network weights
                rover_poi_distances = [[] for i in range(p["n_poi"])]
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                    policy_id = int(pops["EA{0}".format(rd.rovers[rk].self_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rd.rovers[rk].self_id)].population["pol{0}".format(policy_id)]
                    rd.rovers[rk].get_weights(weights)

                poi_rewards = np.zeros((p["n_poi"], p["steps"]))
                for step_id in range(p["steps"]):
                    # Rovers scan environment to collect state information
                    for rk in rd.rovers:
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)
                        rover_poi_distances[rd.pois[poi].poi_id].append(rd.pois[poi].observer_distances)

                    # Calculate rewards at current time step
                    step_rewards = rd.calc_global()
                    for poi_id in range(p["n_poi"]):
                        poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                    # Each rover acts according to policy
                    for rk in rd.rovers:
                        rd.rovers[rk].step(rd.world_x, rd.world_y)

                # Update fitness of policies using reward information
                g_reward = 0
                for poi_id in range(p["n_poi"]):
                    g_reward += max(poi_rewards[poi_id])
                d_rewards = calc_difference(rd.pois, g_reward, rover_poi_distances)
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    pops["EA{0}".format(rover_id)].fitness[policy_id] = d_rewards[rover_id]

            # Testing Phase (test best agent team found so far) ------------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(rd, pops))
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
        end_time = time.time()
        run_times.append(end_time - start_time)

    create_csv_file(run_times, "Output_Data/", "DifferenceRunTimes.csv")


def rover_dpp():
    """
    Train rovers in tightly coupled rover domain using D++
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for CCEA populations
    pops = {}
    for rover_id in range(p["n_rovers"]):
        pops["EA{0}".format(rover_id)] = Ccea(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    run_times = []
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:  # Perform statistical runs
        start_time = time.time()

        # Create new population of policies for each rover
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            for pkey in pops:
                pops[pkey].select_policy_teams()

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(p["pop_size"]):
                rover_poi_distances = [[] for i in range(p["n_poi"])]
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                    policy_id = int(pops["EA{0}".format(rd.rovers[rk].self_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rd.rovers[rk].self_id)].population["pol{0}".format(policy_id)]
                    rd.rovers[rk].get_weights(weights)

                poi_rewards = np.zeros((p["n_poi"], p["steps"]))
                for step_id in range(p["steps"]):
                    # Rovers scan environment to collect state information
                    for rk in rd.rovers:
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)
                        rover_poi_distances[rd.pois[poi].poi_id].append(rd.pois[poi].observer_distances)

                    # Calculate rewards at current time step
                    step_rewards = rd.calc_global()
                    for poi_id in range(p["n_poi"]):
                        poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                    # Each rover acts according to policy
                    for rk in rd.rovers:
                        rd.rovers[rk].step(rd.world_x, rd.world_y)

                # Update fitness of policies using reward information
                g_reward = 0
                for poi_id in range(p["n_poi"]):
                    g_reward += max(poi_rewards[poi_id])
                dpp_rewards = calc_dpp(rd.pois, g_reward, rover_poi_distances)
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    pops["EA{0}".format(rover_id)].fitness[policy_id] = dpp_rewards[rover_id]

            # Testing Phase (test best agent team found so far) ------------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(rd, pops))
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
        end_time = time.time()
        run_times.append(end_time - start_time)

    create_csv_file(run_times, "Output_Data/", "DPPRunTimes.csv")
