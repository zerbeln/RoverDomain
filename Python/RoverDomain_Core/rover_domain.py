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
        self.rover_poi_distances = [[] for i in range(self.n_pois)]  # Tracks rover distances to POI at each time step

        # Rover Instances
        self.rovers = {}  # Dictionary containing instances of rover objects

        # POI Instances
        self.pois = {}  # Dictionary containing instances of PoI objects

    def reset_world(self):
        """
        Reset world to initial conditions.
        """
        self.rover_poi_distances = [[] for i in range(self.n_pois)]
        for rv in self.rovers:
            self.rovers[rv].reset_rover()
        for poi in self.pois:
            self.pois[poi].reset_poi()

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
        Calculate the global reward for the current state as the reward given by each POI.
        :return: Array capturing reward given from each POI at current time step
        """
        global_reward = np.zeros(self.n_pois)

        for poi in self.pois:
            observer_count = 0
            rover_distances = copy.deepcopy(self.pois[poi].observer_distances)
            rover_distances = np.sort(rover_distances)  # Arranges distances from least to greatest

            for rv in range(self.n_rovers):
                if rover_distances[rv] < self.obs_radius:
                    observer_count += 1

            # Update global reward if POI is observed
            if observer_count >= int(self.pois[poi].coupling):
                summed_dist = sum(rover_distances[0:int(self.pois[poi].coupling)])
                global_reward[self.pois[poi].poi_id] = self.pois[poi].value / (summed_dist/self.pois[poi].coupling)

        return global_reward

    def load_poi_configuration(self):
        """
        Load POI configuration from a CSV file
        """
        config_input = []
        with open('./World_Config/POI_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for poi_id in range(self.n_pois):
            poi_x = float(config_input[poi_id][0])
            poi_y = float(config_input[poi_id][1])
            poi_val = float(config_input[poi_id][2])
            poi_coupling = float(config_input[poi_id][3])

            self.pois["P{0}".format(poi_id)] = Poi(poi_x, poi_y, poi_val, poi_coupling, poi_id)

    def load_rover_configuration(self):
        """
        Load Rover configuration from a saved csv file
        """
        config_input = []
        with open('./World_Config/Rover_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for rover_id in range(self.n_rovers):
            rov_x = float(config_input[rover_id][0])
            rov_y = float(config_input[rover_id][1])
            rov_theta = float(config_input[rover_id][2])

            self.rovers["R{0}".format(rover_id)] = Rover(rover_id, rov_x, rov_y, rov_theta)

    def step(self, rover_actions):
        """
        Environment takes in rover actions and returns next state and the global reward
        :param rover_actions: array containing actions for each rover taken from rover neural networks
        :return global_reward: numpy array containing the reward given by each POI for the given state
        """

        # Rovers take action from neural network
        for rv in self.rovers:
            dx = 2 * p["dmax"] * (rover_actions[self.rovers[rv].rover_id][0] - 0.5)
            dy = 2 * p["dmax"] * (rover_actions[self.rovers[rv].rover_id][1] - 0.5)

            # Calculate new rover X Position
            x = dx + self.rovers[rv].loc[0]

            # Rovers cannot move beyond boundaries of the world
            if x < 0:
                x = 0
            elif x > self.world_x - 1:
                x = self.world_x - 1

            # Calculate new rover Y Position
            y = dy + self.rovers[rv].loc[1]

            # Rovers cannot move beyond boundaries of the world
            if y < 0:
                y = 0
            elif y > self.world_y - 1:
                y = self.world_y - 1

            # Update rover position
            self.rovers[rv].loc[0] = x
            self.rovers[rv].loc[1] = y

        # Rovers observe the new world state
        for rv in self.rovers:
            self.rovers[rv].scan_environment(self.rovers, self.pois)
        # Record observer distances for given time step (for reward calculations)
        for poi in self.pois:
            self.pois[poi].update_observer_distances(self.rovers)
            self.rover_poi_distances[self.pois[poi].poi_id].append(self.pois[poi].observer_distances)

        global_reward = self.calc_global()

        return global_reward
