import numpy as np
import math
import sys


class Rover:
    def __init__(self, p, rov_id, rover_pos):
        self.sensor_range = p["obs_rad"]
        self.min_dist = p["min_dist"]
        self.sensor_readings = np.zeros(p["n_inputs"])
        self.self_id = rov_id  # Rover unique identifier
        self.max_steps = p["n_steps"]
        self.angle_res = p["angle_resolution"]
        self.sensor_type = p["sensor_type"]

        # Agent Neuro-Controller Parameters
        self.n_inputs = p["n_inputs"]
        self.n_outputs = p["n_outputs"]
        self.n_hnodes = p["n_hnodes"]  # Number of nodes in hidden layer
        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs)), [self.n_outputs, 1])

        # Rover intiial conditions
        self.rx_init = rover_pos[0]
        self.ry_init = rover_pos[1]
        self.rt_init = rover_pos[2]

        # Current rover coordinates
        self.rover_x = rover_pos[0]
        self.rover_y = rover_pos[1]
        self.rover_theta = rover_pos[2]

    def reset_rover(self):
        """
        Resets the rover to its initial starting conditions (used to reset each training episode)
        :return:
        """
        # Return rover to initial configuration
        self.rover_x = self.rx_init
        self.rover_y = self.ry_init
        self.rover_theta = self.rt_init

        # Reset Neural Network
        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs)), [self.n_outputs, 1])

    def step(self, x_lim, y_lim):
        """
        Rover executes current actions provided by neuro-controller
        :param x_lim: Outter x-limit of the environment
        :param y_lim:  Outter y-limit of the environment
        :return:
        """
        # Get outputs from neuro-controller
        self.get_inputs()
        self.get_outputs()
        rover_action = self.output_layer.copy()
        rover_action = np.clip(rover_action, -1.0, 1.0)

        # Update rover positions based on outputs and assign to dummy variables
        x = rover_action[0, 0] + self.rover_x
        y = rover_action[1, 0] + self.rover_y
        theta = math.atan(y/x) * (180.0/math.pi)

        # Keep theta between 0 and 360 degrees
        while theta < 0.0:
            theta += 360.0
        while theta > 360.0:
            theta -= 360.0
        if math.isnan(theta):
            theta = 0.0

        # Update rover position if rover is within bounds
        if 0.0 <= x < x_lim and 0.0 <= y < y_lim:
            self.rover_x = x
            self.rover_y = y
        self.rover_theta = theta

    def run_rover_scan(self, rovers, n_rovers):
        """
        Rover activates scanner to detect other rovers within the environment
        :param rovers: Dictionary containing rover positions
        :param num_rovers: Parameter designating the number of rovers in the simulation
        :return: Portion of the state vector created from rover scanner
        """
        rover_state = np.zeros(int(360.0 / self.angle_res))
        temp_rover_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]

        # Log rover distances into brackets
        for rover_id in range(n_rovers):
            if self.self_id == rover_id:  # Ignore self
                continue
            rov_x = rovers["AG{0}".format(rover_id)].rover_x
            rov_y = rovers["AG{0}".format(rover_id)].rover_y

            angle, dist = self.get_angle_dist(self.rover_x, self.rover_y, rov_x, rov_y)

            # Clip distance to not overwhelm activation function in NN
            if dist < self.min_dist:
                dist = self.min_dist

            bracket = int(angle / self.angle_res)
            temp_rover_dist_list[bracket].append(1 / dist)

            # Encode Rover information into the state vector
            num_rovers_bracket = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
            if num_rovers_bracket > 0:
                if self.sensor_type == 'density':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_rovers_bracket  # Density Sensor
                elif self.sensor_type == 'summed':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
                elif self.sensor_type == 'closest':
                    rover_state[bracket] = max(temp_rover_dist_list[bracket])  # Closest Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                rover_state[bracket] = -1.0

        return rover_state

    def run_poi_scan(self, poi_info, n_poi):
        """
        Rover queries scanner that detects POIs
        :param pois: multi-dimensional numpy array containing coordinates and values of POIs
        :param num_poi: parameter designating the number of POI in the simulation
        :return: Portion of state-vector constructed from POI scanner
        """
        poi_state = np.zeros(int(360.0 / self.angle_res))
        temp_poi_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]

        # Log POI distances into brackets
        for poi_id in range(n_poi):
            poi_x = poi_info[poi_id, 0]
            poi_y = poi_info[poi_id, 1]
            poi_value = poi_info[poi_id, 2]

            angle, dist = self.get_angle_dist(self.rover_x, self.rover_y, poi_x, poi_y)

            # Clip distance to not overwhelm activation function in NN
            if dist < self.min_dist:
                dist = self.min_dist

            bracket = int(angle / self.angle_res)
            temp_poi_dist_list[bracket].append(poi_value / dist)

        # Encode POI information into the state vector
        for bracket in range(int(360 / self.angle_res)):
            num_poi_bracket = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
            if num_poi_bracket > 0:
                if self.sensor_type == 'density':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_bracket  # Density Sensor
                elif self.sensor_type == 'summed':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
                elif self.sensor_type == 'closest':
                    poi_state[bracket] = max(temp_poi_dist_list[bracket])  # Closest Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                poi_state[bracket] = -1.0

        return poi_state

    def rover_sensor_scan(self, rovers, pois, n_rovers, n_poi):
        """
        Rovers construct a state input vector for the neuro-controller by accessing data from sensors
        :param rovers: Dictionary containing coordinates of rovers
        :param pois: Multi-dimensional numpy array containing POI locations and values
        :param num_rovers: The number of rovers in the simulation
        :param num_poi: The number of POIs in the simulation
        :return:
        """

        poi_state = self.run_poi_scan(pois, n_poi)
        rover_state = self.run_rover_scan(rovers, n_rovers)

        for bracket in range(4):
            self.sensor_readings[bracket] = poi_state[bracket]
            self.sensor_readings[bracket + 4] = rover_state[bracket]

    def get_angle_dist(self, rovx, rovy, x, y):
        """
        Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        :param rovx: X-Position of rover
        :param rovy: Y-Position of rover
        :param x: X-Position of detected object
        :param y: Y-Position of detected object
        :return: angle, dist
        """

        vx = x - rovx
        vy = y - rovy
        angle = math.atan(vy/vx)*(180.0/math.pi)
        angle -= self.rover_theta

        while angle < 0.0:
            angle += 360.0
        while angle > 360.0:
            angle -= 360.0
        if math.isnan(angle):
            angle = 0.0

        # dist = math.sqrt((vx**2) + (vy**2))
        dist = (vx**2) + (vy**2)

        return angle, dist

    def get_inputs(self):  # Get inputs from state-vector
        """
        Transfer state information to the neuro-controller
        :return:
        """

        for i in range(self.n_inputs):
            self.input_layer[i, 0] = self.sensor_readings[i]

    def get_network_weights(self, nn_weights):
        """
        Apply chosen network weights to the agent's neuro-controller
        :param nn_weights: Dictionary of network weights received from the CCEA
        :return:
        """
        self.weights["Layer1"] = np.reshape(np.mat(nn_weights["L1"]), [self.n_hnodes, self.n_inputs])
        self.weights["Layer2"] = np.reshape(np.mat(nn_weights["L2"]), [self.n_outputs, self.n_hnodes])
        self.weights["input_bias"] = np.reshape(np.mat(nn_weights["b1"]), [self.n_hnodes, 1])
        self.weights["hidden_bias"] = np.reshape(np.mat(nn_weights["b2"]), [self.n_outputs, 1])

    def get_outputs(self):
        """
        Run NN to generate outputs
        :return:
        """
        self.hidden_layer = np.dot(self.weights["Layer1"], self.input_layer) + self.weights["input_bias"]
        for i in range(self.n_hnodes):
            self.hidden_layer[i, 0] = self.tanh(self.hidden_layer[i, 0])

        self.output_layer = np.dot(self.weights["Layer2"], self.hidden_layer) + self.weights["hidden_bias"]
        for i in range(self.n_outputs):
            self.output_layer[i, 0] = self.tanh(self.output_layer[i, 0])

    def tanh(self, inp):  # Tanh function as activation function
        """
        tanh neural network activation function
        :param inp: Node value before activation
        :return: Node value after activation
        """
        tanh = (2 / (1 + np.exp(-2 * inp))) - 1

        return tanh

    def sigmoid(self, inp):  # Sigmoid function as activation function
        """
        sigmoid neural network activation function
        :param inp: Node value before activation
        :return: Node value after activation
        """
        sig = 1 / (1 + np.exp(-inp))

        return sig
