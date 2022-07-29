import numpy as np
import sys
from parameters import parameters as p
from global_functions import get_linear_dist, get_squared_dist, get_angle


class Poi:
    def __init__(self, px, py, p_val, pc, p_id):
        self.poi_id = p_id  # POI Identifier
        self.loc = [px, py]  # Location of the POI
        self.value = p_val  # POI Value
        self.coupling = pc  # POI coupling requirement
        self.observer_distances = np.zeros(p["n_rovers"])  # Keeps track of rover distances
        self.observed = False  # Boolean that indicates if a POI is successfully observed

    def reset_poi(self):
        self.observer_distances = np.zeros(p["n_rovers"])
        self.observed = False

    def update_observer_distances(self, rovers):
        for rov in rovers:
            dist = get_linear_dist(rovers[rov].loc[0], rovers[rov].loc[1], self.loc[0], self.loc[1])
            self.observer_distances[rovers[rov].rover_id] = dist


class Rover:
    def __init__(self, rov_id, rov_x, rov_y, rov_theta):
        # Rover Parameters -----------------------------------------------------------------------------------
        self.rover_id = rov_id  # Rover identifier
        self.loc = [rov_x, rov_y, rov_theta]  # Rover location
        self.initial_pos = [rov_x, rov_y, rov_theta]  # Keeps initial position of rover stored for quick reset
        self.dmax = p["dmax"]  # Maximum distance a rover can move each time step

        # Rover Sensor Characteristics -----------------------------------------------------------------------
        self.sensor_type = p["sensor_model"]  # Type of sensors rover is equipped with
        self.sensor_range = None  # Distance rovers can perceive environment (default is infinite)
        self.sensor_res = p["angle_res"]  # Angular resolution of the sensors
        self.n_inputs = p["n_inp"]  # Number of inputs for rover's neural network

        # Rover Data -----------------------------------------------------------------------------------------
        self.observations = np.zeros(p["n_inp"], dtype=np.float128)  # Number of sensor inputs for Neural Network
        self.rover_actions = np.zeros(p["n_out"], dtype=np.float128)  # Motor actions from neural network outputs

    def reset_rover(self):
        """
        Resets the rover to its initial position in the world
        """
        self.loc = self.initial_pos.copy()
        self.observations = np.zeros(self.n_inputs, dtype=np.float128)

    def scan_environment(self, rovers, pois):
        """
        Constructs the state information that gets passed to the rover's neuro-controller
        """
        n_brackets = int(360.0 / self.sensor_res)
        poi_state = self.poi_scan(pois, n_brackets)
        rover_state = self.rover_scan(rovers, n_brackets)

        for i in range(n_brackets):
            self.observations[i] = poi_state[i]
            self.observations[n_brackets + i] = rover_state[i]

    def poi_scan(self, pois, n_brackets):
        """
        Rover queries scanner that detects POIs
        """
        poi_state = np.zeros(n_brackets)
        temp_poi_dist_list = [[] for _ in range(n_brackets)]

        # Log POI distances into brackets
        poi_id = 0
        for pk in pois:
            angle = get_angle(pois[pk].loc[0], pois[pk].loc[1], (p["x_dim"]/2), (p["y_dim"]/2))
            dist = get_squared_dist(pois[pk].loc[0], pois[pk].loc[1], self.loc[0], self.loc[1])

            bracket = int(angle / self.sensor_res)
            if bracket > n_brackets-1:
                bracket -= n_brackets
            temp_poi_dist_list[bracket].append(pois[pk].value / dist)
            poi_id += 1

        # Encode POI information into the state vector
        for bracket in range(n_brackets):
            num_poi_bracket = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
            if num_poi_bracket > 0:
                if self.sensor_type == 'density':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_bracket  # Density Sensor
                elif self.sensor_type == 'summed':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                poi_state[bracket] = -1.0

        return poi_state

    def rover_scan(self, rovers, n_brackets):
        """
        Rover activates scanner to detect other rovers within the environment
        """
        rover_state = np.zeros(n_brackets)
        temp_rover_dist_list = [[] for _ in range(n_brackets)]

        # Log rover distances into brackets
        for rv in rovers:
            if self.rover_id != rovers[rv].rover_id:  # Ignore self
                rov_x = rovers[rv].loc[0]
                rov_y = rovers[rv].loc[1]

                angle = get_angle(rov_x, rov_y, p["x_dim"]/2, p["y_dim"]/2)
                dist = get_squared_dist(rov_x, rov_y, self.loc[0], self.loc[1])

                bracket = int(angle / self.sensor_res)
                if bracket > n_brackets-1:
                    bracket -= n_brackets
                temp_rover_dist_list[bracket].append(1 / dist)

        # Encode Rover information into the state vector
        for bracket in range(n_brackets):
            num_rovers_bracket = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
            if num_rovers_bracket > 0:
                if self.sensor_type == 'density':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_rovers_bracket  # Density Sensor
                elif self.sensor_type == 'summed':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                rover_state[bracket] = -1.0

        return rover_state



