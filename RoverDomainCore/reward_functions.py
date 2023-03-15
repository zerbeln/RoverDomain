import numpy as np
import copy
from parameters import parameters as p


# DIFFERENCE REWARD --------------------------------------------------------------------------------------------------
def calc_difference(pois, global_reward, rov_poi_dist):
    """
    Calculate each rover's difference reward for the current episode
    """
    difference_rewards = np.zeros(p["n_rovers"])
    for agent_id in range(p["n_rovers"]):
        counterfactual_global_reward = 0.0
        for poi in pois:  # For each POI
            poi_reward = 0.0  # Track best POI reward over all time steps for given POI
            for step in range(p["steps"]):
                observer_count = 0
                rover_distances = copy.deepcopy(rov_poi_dist[pois[poi].poi_id][step])
                rover_distances[agent_id] = 1000.00  # Replace Rover action with counterfactual action
                sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                # Check if required observers within range of POI
                for i in range(int(pois[poi].coupling)):
                    if sorted_distances[i] < p["observation_radius"]:
                        observer_count += 1

                # Calculate reward for given POI at current time step
                if observer_count >= int(pois[poi].coupling):
                    summed_observer_distances = sum(sorted_distances[0:int(pois[poi].coupling)])
                    reward = pois[poi].value/(summed_observer_distances/pois[poi].coupling)
                    if reward > poi_reward:
                        poi_reward = reward

            # Update Counterfactual G
            counterfactual_global_reward += poi_reward

        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards


# D++ REWARD ----------------------------------------------------------------------------------------------------------
def calc_dpp(pois, global_reward, rov_poi_dist):
    """
    Calculate D++ rewards for each rover
    """
    d_rewards = calc_difference(pois, global_reward, rov_poi_dist)
    rewards = np.zeros(p["n_rovers"])  # This is just a temporary reward tracker for iterations of counterfactuals
    dpp_rewards = np.zeros(p["n_rovers"])

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    for agent_id in range(p["n_rovers"]):
        counterfactual_global_reward = 0.0
        n_counters = p["n_rovers"]-1
        for poi in pois:
            poi_reward = 0.0  # Track best POI reward over all time steps for given POI
            for step in range(p["steps"]):
                observer_count = 0
                rover_distances = copy.deepcopy(rov_poi_dist[pois[poi].poi_id][step])
                counterfactual_rovers = np.ones(int(n_counters)) * rover_distances[agent_id]
                rover_distances = np.append(rover_distances, counterfactual_rovers)
                sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                # Check if required observers within range of POI
                for i in range(int(pois[poi].coupling)):
                    if sorted_distances[i] < p["observation_radius"]:
                        observer_count += 1

                # Calculate reward for given POI at current time step
                if observer_count >= pois[poi].coupling:
                    summed_observer_distances = sum(sorted_distances[0:int(pois[poi].coupling)])
                    reward = pois[poi].value/(summed_observer_distances/pois[poi].coupling)
                    if reward > poi_reward:
                        poi_reward = reward

            # Update Counterfactual G
            counterfactual_global_reward += poi_reward

        rewards[agent_id] = (counterfactual_global_reward - global_reward)/n_counters

    for agent_id in range(p["n_rovers"]):
        # Compare D++ to D, and iterate through n counterfactuals if D++ > D
        if rewards[agent_id] > d_rewards[agent_id]:
            n_counters = 1
            while n_counters < p["n_rovers"]:
                counterfactual_global_reward = 0.0
                for poi in pois:
                    observer_count = 0
                    poi_reward = 0.0  # Track best POI reward over all time steps for given POI
                    for step in range(p["steps"]):
                        rover_distances = copy.deepcopy(rov_poi_dist[pois[poi].poi_id][step])
                        counterfactual_rovers = np.ones(int(n_counters)) * rover_distances[agent_id]
                        rover_distances = np.append(rover_distances, counterfactual_rovers)
                        sorted_distances = np.sort(rover_distances)  # Sort from least to greatest

                        # Check if required observers within range of POI
                        for i in range(int(pois[poi].coupling)):
                            if sorted_distances[i] < p["observation_radius"]:
                                observer_count += 1

                        # Calculate reward for given POI at current time step
                        if observer_count >= pois[poi].coupling:
                            summed_observer_distances = sum(sorted_distances[0:int(pois[poi].coupling)])
                            reward = pois[poi].value/(summed_observer_distances/pois[poi].coupling)
                            if reward > poi_reward:
                                poi_reward = reward

                    # Update Counterfactual G
                    counterfactual_global_reward += poi_reward

                # Calculate D++ reward with n counterfactuals added
                temp_dpp = (counterfactual_global_reward - global_reward)/n_counters
                if temp_dpp > rewards[agent_id]:
                    rewards[agent_id] = temp_dpp
                    n_counters = p["n_rovers"] + 1  # Stop iterating
                else:
                    n_counters += 1

            dpp_rewards[agent_id] = rewards[agent_id]  # Returns D++ reward for this agent
        else:
            dpp_rewards[agent_id] = d_rewards[agent_id]  # Returns difference reward for this agent

    return dpp_rewards

