import numpy as np
import csv
import copy
from parameters import parameters as p
from RoverDomain_Core.agent import Poi, Rover


class RoverDomain:
    def __init__(self):
        # World attributes
        self.world_x = p["x_dim"]
        self.world_y = p["y_dim"]
        self.n_pois = p["n_poi"]
        self.n_rovers = p["n_rovers"]
        self.obs_radius = p["observation_radius"]  # Maximum distance rovers can make observations of POI at

        # Rover Instances
        self.rovers = {}  # Dictionary containing instances of rover objects

        # POI Instances
        self.pois = {}  # Dictionary containing instances of PoI objects

    def load_world(self):
        """
        Load a rover domain from a saved csv file.
        """
        # Initialize POI positions and values
        self.load_poi_configuration()

        # Initialize Rover Positions
        self.load_rover_configuration()

    def calc_global(self):
        """
        Calculate the global reward at the current time step.
        :return: Array capturing reward given from each POI at current time step
        """
        global_reward = np.zeros(self.n_pois)

        for pk in self.pois:
            observer_count = 0
            rover_distances = copy.deepcopy(self.pois[pk].observer_distances)
            rover_distances = np.sort(rover_distances)  # Arranges distances from least to greatest

            for rov in range(self.n_rovers):
                dist = rover_distances[rov]
                if dist < self.obs_radius:
                    observer_count += 1

            # Update global reward if POI is observed
            if observer_count >= int(self.pois[pk].coupling):
                summed_dist = sum(rover_distances[0:int(self.pois[pk].coupling)])
                global_reward[self.pois[pk].poi_id] = self.pois[pk].value / (summed_dist/self.pois[pk].coupling)

        return global_reward

    def load_poi_configuration(self):
        """
        Load POI configuration from a CSV file
        """
        config_input = []
        with open('World_Config/POI_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for poi_id in range(self.n_pois):
            poi_x = float(config_input[poi_id][0])
            poi_y = float(config_input[poi_id][1])
            poi_val = float(config_input[poi_id][2])
            poi_coupling = float(config_input[poi_id][3])
            poi_quadrant = float(config_input[poi_id][4])

            self.pois["P{0}".format(poi_id)] = Poi(poi_x, poi_y, poi_val, poi_coupling, poi_quadrant, poi_id)

    def load_rover_configuration(self):
        """
        Load Rover configuration from a saved csv file
        """
        config_input = []
        with open('World_Config/Rover_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for rover_id in range(self.n_rovers):
            rov_x = float(config_input[rover_id][0])
            rov_y = float(config_input[rover_id][1])
            rov_theta = float(config_input[rover_id][2])

            self.rovers["R{0}".format(rover_id)] = Rover(rover_id, rov_x, rov_y, rov_theta)

