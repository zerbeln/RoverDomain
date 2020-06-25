from ccea import Ccea
from reward_functions import calc_global_reward, calc_difference_reward, calc_dpp_reward
from rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
from agent import Rover
import csv; import os; import sys
import numpy as np
import warnings


def get_parameters():
    """
    Create dictionary of parameters needed for simulation
    :return:
    """
    parameters = {}

    # Test Parameters
    parameters["s_runs"] = 1  # Number of statistical runs to perform
    parameters["new_world_config"] = 1  # 1 = Create new environment, 0 = Use existing environment
    parameters["running"] = 1  # 1 keeps visualizer from closing (use 0 for multiple stat runs)

    # Neural Network Parameters
    parameters["n_inputs"] = 8
    parameters["n_hnodes"] = 10
    parameters["n_outputs"] = 2

    # CCEA Parameters
    parameters["pop_size"] = 50
    parameters["m_rate"] = 0.1
    parameters["m_prob"] = 0.1
    parameters["epsilon"] = 0.1
    parameters["generations"] = 500
    parameters["n_elites"] = 10

    # Rover Domain Parameters
    parameters["n_rovers"] = 4
    parameters["n_poi"] = 4
    parameters["n_steps"] = 25
    parameters["min_dist"] = 1.0  # Used to clip distance to prevent activation function overload
    parameters["obs_rad"] = 4.0  # Minimum distance at which rovers can make observations of POIs
    parameters["c_req"] = 1  # Number of rovers required to complete an observation
    parameters["x_dim"] = 30.0  # Outer x-limit of the environment
    parameters["y_dim"] = 30.0  # Outer y-limit of the environment
    parameters["angle_resolution"] = 90  # Degree arc which scanner scans through (default is 90)
    parameters["sensor_type"] = 'summed'  # Available types: 'density', 'summed', 'closest'

    return parameters


def save_reward_history(reward_history, file_name):
    """
    Saves the reward history of the rover teams to create plots for learning performance
    :param reward_history:
    :param file_name:
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def save_rover_path(p, rover_path):  # Save path rovers take using best policy found
    """
    Records the path each rover takes using best policy from CCEA (used by visualizer)
    :param p:  parameter dict
    :param rover_path:  Numpy array containing the trajectory of each rover
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files

    rpath_name = os.path.join(dir_name, 'Rover_Paths.txt')

    rpath = open(rpath_name, 'a')
    for rov_id in range(p["n_rovers"]):
        for t in range(p["n_steps"]+1):
            rpath.write('%f' % rover_path[t, rov_id, 0])
            rpath.write('\t')
            rpath.write('%f' % rover_path[t, rov_id, 1])
            rpath.write('\t')
        rpath.write('\n')
    rpath.write('\n')
    rpath.close()


def rovers_global_only(reward_type):
    """
    Train rovers using the global reward
    :param reward_type:
    :return:
    """
    p = get_parameters()
    rd = RoverDomain(p)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rv = {}
    for rv_id in range(p["n_rovers"]):
        rv["AG{0}".format(rv_id)] = Rover(p, rv_id, rd.rover_positions[rv_id])
        rv["EA{0}".format(rv_id)] = Ccea(p)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["c_req"])

    for srun in range(p["s_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        for rv_id in range(p["n_rovers"]):
            rv["EA{0}".format(rv_id)].create_new_population()
        reward_history = []

        for gen in range(p["generations"]):
            for rv_id in range(p["n_rovers"]):
                rv["EA{0}".format(rv_id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].reset_rover()  # Reset rover to initial conditions
                    policy_id = int(rv["EA{0}".format(rv_id)].team_selection[team_number])
                    weights = rv["EA{0}".format(rv_id)].population["pop{0}".format(policy_id)]
                    rv["AG{0}".format(rv_id)].get_network_weights(weights)  # Apply network weights from CCEA
                    rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, -1)  # Record starting position of each rover

                for step_id in range(p["n_steps"]):
                    # Rover scans environment and constructs state vector
                    for rv_id in range(p["n_rovers"]):
                        rv["AG{0}".format(rv_id)].rover_sensor_scan(rv, rd.pois, p["n_rovers"], p["n_poi"])

                    # Rover processes scan information and acts
                    for rv_id in range(p["n_rovers"]):
                        rv["AG{0}".format(rv_id)].step(p["x_dim"], p["y_dim"])
                        rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, step_id)

                # Update fitness of policies using reward information
                global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
                for rv_id in range(p["n_rovers"]):
                    policy_id = int(rv["EA{0}".format(rv_id)].team_selection[team_number])
                    rv["EA{0}".format(rv_id)].fitness[policy_id] = global_reward

            # Testing Phase (test best policies found so far)
            rd.clear_rover_path()
            for rv_id in range(p["n_rovers"]):
                rv["AG{0}".format(rv_id)].reset_rover()  # Reset rover to initial conditions
                policy_id = np.argmax(rv["EA{0}".format(rv_id)].fitness)
                weights = rv["EA{0}".format(rv_id)].population["pop{0}".format(policy_id)]
                rv["AG{0}".format(rv_id)].get_network_weights(weights)  # Apply best set of weights to network
                rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, -1)

            for step_id in range(p["n_steps"]):
                # Rover scans environment and constructs state vector
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].rover_sensor_scan(rv, rd.pois, p["n_rovers"], p["n_poi"])

                # Rover processes scan information and acts
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].step(p["x_dim"], p["y_dim"])
                    rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, step_id)

            global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                save_rover_path(p, rd.rover_path)

            # Choose new parents and create new offspring population
            for rv_id in range(p["n_rovers"]):
                rv["EA{0}".format(rv_id)].down_select()

        save_reward_history(reward_history, "Global_Reward.csv")

    run_visualizer(p)


def rovers_difference_rewards(reward_type):
    """
    Train rovers using their difference reward
    :param reward_type:
    :return:
    """
    p = get_parameters()
    rd = RoverDomain(p)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rv = {}
    for rv_id in range(p["n_rovers"]):
        rv["AG{0}".format(rv_id)] = Rover(p, rv_id, rd.rover_positions[rv_id])
        rv["EA{0}".format(rv_id)] = Ccea(p)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["c_req"])

    for srun in range(p["s_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        for rv_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rv["EA{0}".format(rv_id)].create_new_population()
        reward_history = []

        for gen in range(p["generations"]):
            for rv_id in range(p["n_rovers"]):
                rv["EA{0}".format(rv_id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].reset_rover()  # Reset rover to initial conditions
                    policy_id = int(rv["EA{0}".format(rv_id)].team_selection[team_number])
                    weights = rv["EA{0}".format(rv_id)].population["pop{0}".format(policy_id)]
                    rv["AG{0}".format(rv_id)].get_network_weights(weights)  # Apply network weights from CCEA
                    rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, -1)  # Record starting position of each rover

                for step_id in range(p["n_steps"]):
                    # Rover scans environment and constructs state vector
                    for rv_id in range(p["n_rovers"]):
                        rv["AG{0}".format(rv_id)].rover_sensor_scan(rv, rd.pois, p["n_rovers"], p["n_poi"])

                    # Rover processes scan information and acts
                    for rv_id in range(p["n_rovers"]):
                        rv["AG{0}".format(rv_id)].step(p["x_dim"], p["y_dim"])
                        rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, step_id)

                # Update fitness of policies using reward information
                global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
                difference_rewards = calc_difference_reward(p, rd.rover_path, rd.pois, global_reward)
                for rv_id in range(p["n_rovers"]):
                    policy_id = int(rv["EA{0}".format(rv_id)].team_selection[team_number])
                    rv["EA{0}".format(rv_id)].fitness[policy_id] = difference_rewards[rv_id]

            # Testing Phase (test best policies found so far) ---------------------------------------------------------
            for rv_id in range(p["n_rovers"]):
                rv["AG{0}".format(rv_id)].reset_rover()  # Reset rover to initial conditions
                policy_id = np.argmax(rv["EA{0}".format(rv_id)].fitness)
                weights = rv["EA{0}".format(rv_id)].population["pop{0}".format(policy_id)]
                rv["AG{0}".format(rv_id)].get_network_weights(weights)  # Apply best set of weights to network
                rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, -1)

            for step_id in range(p["n_steps"]):
                # Rover scans environment and constructs state vector
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].rover_sensor_scan(rv, rd.pois, p["n_rovers"], p["n_poi"])

                # Rover processes information from scan and acts
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].step(p["x_dim"], p["y_dim"])
                    rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, step_id)

            global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                save_rover_path(p, rd.rover_path)

            # Choose new parents and create new offspring population
            for rv_id in range(p["n_rovers"]):
                rv["EA{0}".format(rv_id)].down_select()

        save_reward_history(reward_history, "Difference_Reward.csv")
    run_visualizer(p)


def rovers_dpp_rewards(reward_type):
    """
    Train rovers using the D++ reward
    :param reward_type:
    :return:
    """
    p = get_parameters()
    rd = RoverDomain(p)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rv = {}
    for rv_id in range(p["n_rovers"]):
        rv["AG{0}".format(rv_id)] = Rover(p, rv_id, rd.rover_positions[rv_id])
        rv["EA{0}".format(rv_id)] = Ccea(p)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["c_req"])

    for srun in range(p["s_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        for rv_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rv["EA{0}".format(rv_id)].create_new_population()
        reward_history = []

        for gen in range(p["generations"]):
            for rv_id in range(p["n_rovers"]):
                rv["EA{0}".format(rv_id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].reset_rover()  # Reset rover to initial conditions
                    policy_id = int(rv["EA{0}".format(rv_id)].team_selection[team_number])
                    weights = rv["EA{0}".format(rv_id)].population["pop{0}".format(policy_id)]
                    rv["AG{0}".format(rv_id)].get_network_weights(weights)  # Apply network weights from CCEA
                    rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, -1)  # Record starting position of each rover

                for step_id in range(p["n_steps"]):
                    # Rover scans environment and constructs state vector
                    for rv_id in range(p["n_rovers"]):
                        rv["AG{0}".format(rv_id)].rover_sensor_scan(rv, rd.pois, p["n_rovers"], p["n_poi"])

                    # Rover processes scan information and acts
                    for rv_id in range(p["n_rovers"]):
                        rv["AG{0}".format(rv_id)].step(p["x_dim"], p["y_dim"])
                        rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, step_id)

                # Update fitness of policies using reward information
                global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
                dpp_rewards = calc_dpp_reward(p, rd.rover_path, rd.pois, global_reward)
                for rv_id in range(p["n_rovers"]):
                    policy_id = int(rv["EA{0}".format(rv_id)].team_selection[team_number])
                    rv["EA{0}".format(rv_id)].fitness[policy_id] = dpp_rewards[rv_id]

            # Testing Phase (test best policies found so far) ---------------------------------------------------------
            for rv_id in range(p["n_rovers"]):
                rv["AG{0}".format(rv_id)].reset_rover()  # Reset rover to initial conditions
                policy_id = np.argmax(rv["EA{0}".format(rv_id)].fitness)
                weights = rv["EA{0}".format(rv_id)].population["pop{0}".format(policy_id)]
                rv["AG{0}".format(rv_id)].get_network_weights(weights)  # Apply best set of weights to network
                rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, -1)

            for step_id in range(p["n_steps"]):
                # Rover scans environment and constructs state vector
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].rover_sensor_scan(rv, rd.pois, p["n_rovers"], p["n_poi"])

                # Rover processes information from scan and acts
                for rv_id in range(p["n_rovers"]):
                    rv["AG{0}".format(rv_id)].step(p["x_dim"], p["y_dim"])
                    rd.update_rover_path(rv["AG{0}".format(rv_id)], rv_id, step_id)

            global_reward = calc_global_reward(p, rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                save_rover_path(p, rd.rover_path)

            # Choose new parents and create new offspring population
            for rv_id in range(p["n_rovers"]):
                rv["EA{0}".format(rv_id)].down_select()

        save_reward_history(reward_history, "DPP_Reward.csv")
    run_visualizer(p)


def main(reward_type="Global"):
    """
    reward_type: Global, Difference, or DPP
    :return:
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if reward_type == "Global":
        rovers_global_only(reward_type)
    elif reward_type == "Difference":
        rovers_difference_rewards(reward_type)
    elif reward_type == "DPP":
        rovers_dpp_rewards(reward_type)
    else:
        sys.exit('Incorrect Reward Type')


main(reward_type="Global")  # Run the program
