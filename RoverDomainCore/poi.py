from global_functions import get_linear_dist
from parameters import parameters as p
import numpy as np


class POI:
    def __init__(self, px, py, p_val, coupling, poi_id):
        self.poi_id = poi_id  # POI Identifier
        self.loc = [px, py]  # Location of the POI
        self.value = p_val  # POI Value
        self.coupling = coupling  # POI coupling requirement
        self.observer_distances = np.zeros(p["n_rovers"])  # Keeps track of rover distances
        self.observed = False  # Boolean that indicates if a POI is successfully observed
        self.quadrant = None  # Tracks which quadrant (or sector) of the environment a POI exists in

    def reset_poi(self, poi_config):
        """
        Clears the observer distances array and sets POI observed boolean back to False
        """
        self.loc[0] = poi_config[0]
        self.loc[1] = poi_config[1]
        self.value = poi_config[2]
        self.coupling = poi_config[3]
        self.observer_distances = np.zeros(p["n_rovers"])
        self.observed = False

    def update_observer_distances(self, rovers):
        """
        Records the linear distances between rovers in the system and the POI for use in reward calculations
        """
        for rov in rovers:
            dist = get_linear_dist(rovers[rov].loc[0], rovers[rov].loc[1], self.loc[0], self.loc[1])
            self.observer_distances[rovers[rov].rover_id] = dist
