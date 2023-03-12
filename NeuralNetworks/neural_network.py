import numpy as np


class NeuralNetwork:
    def __init__(self, n_inp=8, n_hid=10, n_out=2):
        self.n_inputs = n_inp  # Number of nodes in input layer
        self.n_outputs = n_out  # Number of nodes in output layer
        self.n_hnodes = n_hid  # Number of nodes in hidden layer
        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs, dtype=np.longdouble)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes, dtype=np.longdouble)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs, dtype=np.longdouble)), [self.n_outputs, 1])

    # Rover Control NN ------------------------------------------------------------------------------------------------
    def get_weights(self, nn_weights):
        """
        Apply chosen network weights to the agent's neuro-controller
        """
        self.weights['Layer1'] = np.reshape(np.mat(nn_weights['L1']), [self.n_hnodes, self.n_inputs])
        self.weights['Layer2'] = np.reshape(np.mat(nn_weights['L2']), [self.n_outputs, self.n_hnodes])
        self.weights['input_bias'] = np.reshape(np.mat(nn_weights['b1']), [self.n_hnodes, 1])
        self.weights['hidden_bias'] = np.reshape(np.mat(nn_weights['b2']), [self.n_outputs, 1])

    def get_nn_inputs(self, sensor_data):
        """
        Take in state information collected by rover and assign to input layer of network
        """
        for i in range(self.n_inputs):
            self.input_layer[i, 0] = sensor_data[i]

    def get_nn_outputs(self):
        """
        Run rover NN to generate actions
        """
        self.hidden_layer = np.dot(self.weights['Layer1'], self.input_layer) + self.weights['input_bias']
        self.hidden_layer = self.sigmoid(self.hidden_layer)

        self.output_layer = np.dot(self.weights['Layer2'], self.hidden_layer) + self.weights['hidden_bias']
        self.output_layer = self.sigmoid(self.output_layer)

    def run_rover_nn(self, sensor_data):
        """
        Run neural network using state information, return rover actions
        """
        self.get_nn_inputs(sensor_data)
        self.get_nn_outputs()

        rover_actions = np.zeros(self.n_outputs)
        for i in range(self.n_outputs):
            rover_actions[i] = self.output_layer[i, 0]

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

