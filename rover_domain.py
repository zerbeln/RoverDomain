import numpy as np
import math
import random
import os
import csv


class RoverDomain:

    def __init__(self, p):
        # World attributes
        self.world_x = p["x_dim"]
        self.world_y = p["y_dim"]
        self.n_poi = p["n_poi"]
        self.n_rovers = p["n_rovers"]
        self.c_req = p["c_req"]
        self.min_dist = p["min_dist"]
        self.obs_radius = p["obs_rad"]
        self.create_new_world_config = p["new_world_config"]
        self.rover_steps = p["n_steps"]

        # Rover path trace for trajectory-wide global reward computation and vizualization purposes
        self.rover_path = np.zeros(((self.rover_steps + 1), self.n_rovers, 3))

        # POI position and value vectors
        self.pois = np.zeros((self.n_poi, 3))  # [X, Y, Val]
        self.rover_positions = np.zeros((self.n_rovers, 3))  # [X, Y, Theta]

        # User Defined Parameters:
        self.poi_observations = np.zeros(self.n_poi)  # Used for spatial coupling of POIs

        self.initial_world_setup()

    def initial_world_setup(self):
        """
        Set POI positions and POI values, clear the rover path tracker
        :return: none
        """
        self.pois = np.zeros((self.n_poi, 3))

        if self.create_new_world_config == 1:
            # Initialize Rover positions
            self.init_rover_pos_random_concentrated(self.world_x, self.world_y, radius=5.0)
            self.save_rover_configuration()

            # Initialize POI positions and values
            self.init_poi_pos_four_corners()
            self.init_poi_vals_random()
            self.save_poi_configuration()
        else:
            # Intialize Rover positions
            self.use_saved_rover_config()

            # Initialize POI positions and values
            self.use_saved_poi_configuration()

        self.rover_path = np.zeros(((self.rover_steps + 1), self.n_rovers, 3))  # Tracks rover trajectories

    def clear_rover_path(self):
        """
        Resets the rover path tracker
        :return:
        """
        self.rover_path = np.zeros(((self.rover_steps + 1), self.n_rovers, 3))  # Tracks rover trajectories

    def update_rover_path(self, rover, rover_id, step_id):
        """
        Update the array tracking the path of each rover
        :param rover:  Dictionary containing instances of rovers
        :param rover_id: identifier for an individual rover
        :param step_id: Current time step for the simulation
        :return:
        """
        self.rover_path[step_id+1, rover_id, 0] = rover.rover_x
        self.rover_path[step_id+1, rover_id, 1] = rover.rover_y
        self.rover_path[step_id+1, rover_id, 2] = rover.rover_theta

    # ROVER POSITION FUNCTIONS ----------------------------------------------------------------------------------------
    def use_saved_rover_config(self):
        """
        Use a stored initial configuration from a CSV file
        :return:
        """
        config_input = []
        with open('Output_Data/Rover_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        # Assign values to variables that track the rover's initial conditions
        for rov_id in range(self.n_rovers):
            self.rover_positions[rov_id, 0] = float(config_input[rov_id][0])
            self.rover_positions[rov_id, 1] = float(config_input[rov_id][1])
            self.rover_positions[rov_id, 2] = float(config_input[rov_id][2])

    def save_rover_configuration(self):
        """
        Saves rover positions to a csv file in a folder called Output_Data
        :Output: CSV file containing rover starting positions
        """
        dir_name = 'Output_Data/'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'Rover_Config.csv')

        row = np.zeros(3)
        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for rov_id in range(self.n_rovers):
                row[0] = self.rover_positions[rov_id, 0]
                row[1] = self.rover_positions[rov_id, 1]
                row[2] = self.rover_positions[rov_id, 2]
                writer.writerow(row[:])

    def init_rover_pos_random(self, x_lim, y_lim):  # Randomly set rovers on map
        """
        Rovers given random starting positions and orientations
        :param x_lim: Outter x-limit of the environment
        :param y_lim: Outter y-limit of the environment
        :return:
        """
        for rov_id in range(self.n_rovers):
            self.rover_positions[rov_id, 0] = random.uniform(0.0, x_lim-1.0)
            self.rover_positions[rov_id, 1] = random.uniform(0.0, y_lim-1.0)
            self.rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)

    def init_rover_pos_random_concentrated(self, x_lim, y_lim, radius=4.0):
        """
        Rovers given random starting positions within a radius of the center. Starting orientations are random.
        :param x_lim: Outter x-limit of the environment
        :param y-lim: Outter y-limit of the environment
        :param radius: maximum radius from the center rovers are allowed to start in
        :return:
        """
        # Origin of constraining circle
        center_x = x_lim/2.0
        center_y = y_lim/2.0

        for rov_id in range(self.n_rovers):
            x = random.uniform(0.0, x_lim-1.0)  # Rover X-Coordinate
            y = random.uniform(0.0, y_lim-1.0)  # Rover Y-Coordinate

            # Make sure coordinates are within the bounds of the constraining circle
            while x > (center_x + radius) or x < (center_x - radius):
                x = random.uniform(0.0, x_lim-1.0)
            while y > (center_y + radius) or y < (center_y - radius):
                y = random.uniform(0.0, y_lim-1.0)

            self.rover_positions[rov_id, 0] = x
            self.rover_positions[rov_id, 1] = y
            self.rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)

    # POI POSITION FUNCTIONS ------------------------------------------------------------------------------------------
    def save_poi_configuration(self):
        """
        Saves world configuration to a csv file in a folder called Output_Data
        :Output: One CSV file containing POI postions and POI values
        """
        dir_name = 'Output_Data/'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'POI_Config.csv')

        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for poi_id in range(self.n_poi):
                writer.writerow(self.pois[poi_id, :])

    def use_saved_poi_configuration(self):
        """
        Re-use world configuration stored in a CSV file in folder called Output_Data
        :return:
        """
        config_input = []
        with open('Output_Data/POI_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for poi_id in range(self.n_poi):
            self.pois[poi_id, 0] = float(config_input[poi_id][0])
            self.pois[poi_id, 1] = float(config_input[poi_id][1])
            self.pois[poi_id, 2] = float(config_input[poi_id][2])

    def init_poi_pos_random(self, rovers):  # Randomly set POI on the map
        """
        POI positions set randomly across the map (but not in range of any rover)
        :return:
        """
        for poi_id in range(self.n_poi):
            x = random.uniform(0, self.world_x-1.0)
            y = random.uniform(0, self.world_y-1.0)

            rover_id = 0
            while rover_id < self.n_rovers:
                rovx = rovers[rover_id, 0]
                rovy = rovers[rover_id, 1]
                xdist = x - rovx; ydist = y - rovy
                distance = math.sqrt((xdist**2) + (ydist**2))

                while distance < self.obs_radius:
                    x = random.uniform(0, self.world_x - 1.0)
                    y = random.uniform(0, self.world_y - 1.0)
                    rovx = rovers["AG{0}".format(rover_id)].rover_x
                    rovy = rovers["AG{0}".format(rover_id)].rover_y
                    xdist = x - rovx; ydist = y - rovy
                    distance = math.sqrt((xdist ** 2) + (ydist ** 2))

                    if distance > self.obs_radius:
                        rover_id = -1

                rover_id += 1

            self.pois[poi_id, 0] = x
            self.pois[poi_id, 1] = y

    def init_poi_pos_circle(self):
        """
        POI positions are set in a circle around the center of the map at a specified radius.
        :return:
        """
        radius = 15.0
        interval = float(360/self.n_poi)

        x = self.world_x/2.0
        y = self.world_y/2.0
        theta = 0.0

        for poi_id in range(self.n_poi):
            self.pois[poi_id, 0] = x + radius*math.cos(theta*math.pi/180)
            self.pois[poi_id, 1] = y + radius*math.sin(theta*math.pi/180)
            theta += interval

    def init_poi_pos_two_poi(self):
        """
        Sets two POI on the map, one on each side and aligned with the center
        :return:
        """
        assert(self.n_poi == 2)

        self.pois[0, 0] = 1.0; self.pois[0, 1] = self.world_y/2.0
        self.pois[1, 0] = (self.world_x-2.0); self.pois[1, 1] = self.world_y/2.0

    def init_poi_pos_four_corners(self):  # Statically set 4 POI (one in each corner)
        """
        Sets 4 POI on the map (one in each corner)
        :return:
        """
        assert(self.n_poi == 4)  # There must only be 4 POI for this initialization

        self.pois[0, 0] = 2.0; self.pois[0, 1] = 2.0  # Bottom left
        self.pois[1, 0] = 2.0; self.pois[1, 1] = (self.world_y - 2.0)  # Top left
        self.pois[2, 0] = (self.world_x - 2.0); self.pois[2, 1] = 2.0  # Bottom right
        self.pois[3, 0] = (self.world_x - 2.0); self.pois[3, 1] = (self.world_y - 2.0)  # Top right

    # POI VALUE FUNCTIONS -----------------------------------------------------------------------------------
    def init_poi_vals_random(self):
        """
        POI values randomly assigned 1-10
        :return:
        """
        for poi_id in range(self.n_poi):
            self.pois[poi_id, 2] = float(random.randint(1, 12))

    def init_poi_vals_fixed_ascending(self):
        """
        POI values set to fixed, ascending values based on POI ID
        :return:
        """
        for poi_id in range(self.n_poi):
            self.pois[poi_id, 2] = poi_id + 1

    def init_poi_vals_fixed(self):
        """
        Set all POIs to a static, fixed value
        :return:
        """
        for poi_id in range(self.n_poi):
            self.pois[poi_id, 2] = 1.0
