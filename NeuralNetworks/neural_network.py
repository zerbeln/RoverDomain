import numpy as np


class NeuralNetwork:
    def __init__(self, n_inp=8, n_hid=10, n_out=2):
        self.n_inputs = n_inp  # Number of nodes in input layer
        self.n_outputs = n_out  # Number of nodes in output layer
        self.n_hidden = n_hid  # Number of nodes in hidden layer
        self.weights = {}
        self.input_layer = np.zeros(self.n_inputs)
        self.hidden_layer = np.zeros(self.n_hidden)
        self.output_layer = np.zeros(self.n_outputs)

    # Rover Control NN ------------------------------------------------------------------------------------------------
    def get_weights(self, nn_weights):
        """
        Apply chosen network weights to the agent's neuro-controller
        """
        self.weights['l1'] = nn_weights['L1'][0:self.n_inputs, :]
        self.weights['l1_bias'] = nn_weights['L1'][self.n_inputs, :]  # Biasing weights
        self.weights['l2'] = nn_weights['L2'][0:self.n_hidden, :]
        self.weights['l2_bias'] = nn_weights['L2'][self.n_hidden, :]  # Biasing weights

    def get_nn_inputs(self, sensor_data):
        """
        Take in state information collected by rover and assign to input layer of network
        """
        self.input_layer = sensor_data.copy()

    def get_nn_outputs(self):
        """
        Run rover NN to generate actions
        """
        self.hidden_layer = np.dot(self.input_layer, self.weights['l1']) + self.weights['l1_bias']
        self.hidden_layer = self.sigmoid(self.hidden_layer)

        self.output_layer = np.dot(self.hidden_layer, self.weights['l2']) + self.weights['l2_bias']
        self.output_layer = self.sigmoid(self.output_layer)

    def run_rover_nn(self, sensor_data):
        """
        Run neural network using state information, return rover actions
        """
        self.get_nn_inputs(sensor_data)
        self.get_nn_outputs()

        rover_actions = self.output_layer.copy()

        return rover_actions

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
