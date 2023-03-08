import numpy as np
from NeuralNetworks.neural_network import NeuralNetwork


class SupervisorNetwork(NeuralNetwork):
    def __init__(self, n_inp=8, n_hid=10, n_out=8, n_agents=1):
        super().__init__(n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        self.n_agents = n_agents

    def run_supervisor_nn(self, sensor_data):
        """
        Run neural network using state information, return rover counterfactuals
        :param sensor_data: array containing state information from rover observations of the current state
        :return: array containing rover's next actions
        """
        self.get_nn_inputs(sensor_data)
        self.get_nn_outputs()

        rover_counterfactuals = {}
        for rover_id in range(self.n_agents):
            counterfactual = np.zeros(self.n_inputs)
            for i in range(self.n_inputs):
                counterfactual[i] = self.output_layer[rover_id*self.n_inputs + i, 0]

            rover_counterfactuals["RV{0}".format(rover_id)] = counterfactual.copy()

        return rover_counterfactuals

