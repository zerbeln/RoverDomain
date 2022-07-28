import numpy as np
import math
import sys
from parameters import parameters as p
from global_functions import get_linear_dist, get_squared_dist, get_angle
import warnings
warnings.filterwarnings('ignore')


class Poi:
    def __init__(self, px, py, p_val, pc, pq, p_id):
        self.poi_id = p_id  # POI Identifier
        self.loc = [px, py]  # Location of the POI
        self.value = p_val  # POI Value
        self.coupling = pc  # POI coupling requirement
        self.observer_distances = np.zeros(p["n_rovers"])  # Keeps track of rover distances
        self.observed = False  # Boolean that indicates whether or not a POI is successfully observed

        # User Defined Parameters ---------------------------------------------------------------------------
        self.quadrant = pq  # POI Quadrant location
        self.hazardous = False  # Boolean that indicates whether or not a POI is in a hazard zone

    def reset_poi(self):
        self.observer_distances = np.zeros(p["n_rovers"])

    def update_observer_distances(self, rovers):
        for rov in rovers:
            dist = get_linear_dist(rovers[rov].loc[0], rovers[rov].loc[1], self.loc[0], self.loc[1])
            self.observer_distances[rovers[rov].self_id] = dist


class Rover:
    def __init__(self, rov_id, rov_x, rov_y, rov_theta):
        # Rover Parameters -----------------------------------------------------------------------------------
        self.self_id = rov_id  # Rover identifier
        self.loc = [rov_x, rov_y, rov_theta]  # Rover location
        self.initial_pos = [rov_x, rov_y, rov_theta]  # Keeps initial position of rover stored for quick reset
        self.dmax = p["dmax"]  # Maximum distance a rover can move each time step

        # Rover Sensor Characteristics -----------------------------------------------------------------------
        self.sensor_type = p["sensor_model"]  # Type of sensors rover is equipped with
        self.sensor_range = None  # Distance rovers can perceive environment (default is infinite)
        self.sensor_res = p["angle_res"]  # Angular resolution of the sensors

        # Rover Data -----------------------------------------------------------------------------------------
        self.sensor_readings = np.zeros(p["n_inp"], dtype=np.float128)  # Number of sensor inputs for Neural Network
        self.rover_actions = np.zeros(p["n_out"], dtype=np.float128)  # Motor actions from neural network outputs

        # Rover Motor Controller -----------------------------------------------------------------------------
        self.n_inputs = p["n_inp"]
        self.n_outputs = p["n_out"]
        self.n_hnodes = p["n_hid"]  # Number of nodes in hidden layer
        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs, dtype=np.float128)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes, dtype=np.float128)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs, dtype=np.float128)), [self.n_outputs, 1])

        # Rover CBM --------------------------------------------------------------------------------------------
        self.policy_bank = {}  # Pre-trained set of skills used by rovers

    def reset_rover(self):
        """
        Resets the rover to its initial position in the world
        """
        self.loc = self.initial_pos.copy()
        self.sensor_readings = np.zeros(self.n_inputs, dtype=np.float128)

    def step(self, world_x, world_y):
        """
        Rover executes current actions provided by neuro-controller (not using policy playbook)
        """
        # Get outputs from neuro-controller
        self.get_nn_outputs()

        # Update rover positions based on outputs and assign to dummy variables
        dx = 2 * self.dmax * (self.rover_actions[0] - 0.5)
        dy = 2 * self.dmax * (self.rover_actions[1] - 0.5)

        # Update X Position
        x = dx + self.loc[0]
        if x < 0:
            x = 0
        elif x > world_x-1:
            x = world_x-1

        # Update Y Position
        y = dy + self.loc[1]
        if y < 0:
            y = 0
        elif y > world_y-1:
            y = world_y-1

        self.loc[0] = x
        self.loc[1] = y

    def scan_environment(self, rovers, pois):
        """
        Constructs the state information that gets passed to the rover's neuro-controller
        """
        n_brackets = int(360.0 / self.sensor_res)
        poi_state = self.poi_scan(pois, n_brackets)
        rover_state = self.rover_scan(rovers, n_brackets)

        for i in range(n_brackets):
            self.sensor_readings[i] = poi_state[i]
            self.sensor_readings[n_brackets + i] = rover_state[i]
            self.input_layer[i, 0] = poi_state[i]
            self.input_layer[n_brackets + i, 0] = rover_state[i]

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
        for rk in rovers:
            if self.self_id != rovers[rk].self_id:  # Ignore self
                rov_x = rovers[rk].loc[0]
                rov_y = rovers[rk].loc[1]

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

    # Motor Control NN ------------------------------------------------------------------------------------------------
    def get_weights(self, nn_weights):
        """
        Apply chosen network weights to the agent's neuro-controller
        """
        self.weights["Layer1"] = np.reshape(np.mat(nn_weights["L1"]), [self.n_hnodes, self.n_inputs])
        self.weights["Layer2"] = np.reshape(np.mat(nn_weights["L2"]), [self.n_outputs, self.n_hnodes])
        self.weights["input_bias"] = np.reshape(np.mat(nn_weights["b1"]), [self.n_hnodes, 1])
        self.weights["hidden_bias"] = np.reshape(np.mat(nn_weights["b2"]), [self.n_outputs, 1])

    def get_nn_outputs(self):
        """
        Run rover NN to generate actions
        """
        self.hidden_layer = np.dot(self.weights["Layer1"], self.input_layer) + self.weights["input_bias"]
        self.hidden_layer = self.sigmoid(self.hidden_layer)

        self.output_layer = np.dot(self.weights["Layer2"], self.hidden_layer) + self.weights["hidden_bias"]
        self.output_layer = self.sigmoid(self.output_layer)

        for i in range(self.n_outputs):
            self.rover_actions[i] = self.output_layer[i, 0]

    # Activation Functions -------------------------------------------------------------------------------------------
    def tanh(self, inp):  # Tanh function as activation function
        """
        tanh neural network activation function
        """

        tanh = (2 / (1 + np.exp(-2 * inp))) - 1

        return tanh

    def sigmoid(self, inp):  # Sigmoid function as activation function
        """
        sigmoid neural network activation function
        """

        sig = 1 / (1 + np.exp(-inp))

        return sig

    # User Defined Functions -----------------------------------------------------------------------------------------
    def custom_step(self, world_x, world_y):
        """
        Rover executes current actions provided by neuro-controller (not using policy playbook)
        """

        # Update rover positions based on outputs and assign to dummy variables
        dx = 2 * self.dmax * self.rover_actions[0]
        dy = 2 * self.dmax * self.rover_actions[1]

        # Update X Position
        x = dx + self.loc[0]
        if x < 0:
            x = 0
        elif x > world_x - 1:
            x = world_x - 1

        # Update Y Position
        y = dy + self.loc[1]
        if y < 0:
            y = 0
        elif y > world_y - 1:
            y = world_y - 1

        self.loc[0] = x
        self.loc[1] = y

