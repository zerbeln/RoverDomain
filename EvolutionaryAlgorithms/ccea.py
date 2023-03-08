import numpy as np
import random
from EvolutionaryAlgorithms.ea import EA


class CCEA(EA):
    def __init__(self, n_inp=8, n_hid=10, n_out=2):
        super().__init__(n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        self.team_selection = np.ones(self.pop_size) * (-1)

    def select_policy_teams(self):
        """
        Determines which individuals from each population get paired together
        """

        self.team_selection = random.sample(range(self.pop_size), self.pop_size)