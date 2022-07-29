from RoverDomain_Core.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
import numpy as np
from parameters import parameters as p
from global_functions import create_pickle_file
import random
import math
import csv
import os
from global_functions import get_linear_dist, get_angle


def save_poi_configuration(pois_info):
    """
    Saves POI configuration to a csv file in a folder called World_Config
    """
    dir_name = './World_Config'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    pfile_name = os.path.join(dir_name, 'POI_Config.csv')

    with open(pfile_name, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for poi_id in range(p["n_poi"]):
            writer.writerow(pois_info[poi_id, :])

    csvfile.close()


def save_rover_configuration(initial_rover_positions):
    """
    Saves Rover configuration to a csv file in a folder called World_Config
    """
    dir_name = './World_Config'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    pfile_name = os.path.join(dir_name, 'Rover_Config.csv')

    row = np.zeros(3)
    with open(pfile_name, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for rov_id in range(p["n_rovers"]):
            row[0] = initial_rover_positions[rov_id, 0]
            row[1] = initial_rover_positions[rov_id, 1]
            row[2] = initial_rover_positions[rov_id, 2]
            writer.writerow(row[:])

    csvfile.close()


# ROVER POSITION FUNCTIONS ---------------------------------------------------------------------------------------
def rover_pos_random(pois_info):  # Randomly set rovers on map
    """
    Rovers given random starting positions and orientations. Code ensures rovers do not start out too close to POI.
    """
    initial_rover_positions = np.zeros((p["n_rovers"], 3))

    for rov_id in range(p["n_rovers"]):
        rover_x = random.uniform(0.0, p["x_dim"]-1.0)
        rover_y = random.uniform(0.0, p["y_dim"]-1.0)
        rover_theta = random.uniform(0.0, 360.0)
        buffer = 3  # Smallest distance to the outer POI observation area a rover can spawn

        # Make sure rover does not spawn within observation range of a POI
        rover_too_close = True
        while rover_too_close:
            count = 0
            for poi_id in range(p["n_poi"]):
                dist = get_linear_dist(pois_info[poi_id, 0], pois_info[poi_id, 1], rover_x, rover_y)
                if dist < (p["observation_radius"] + buffer):
                    count += 1

            if count == 0:
                rover_too_close = False
            else:
                rover_x = random.uniform(0.0, p["x_dim"] - 1.0)
                rover_y = random.uniform(0.0, p["y_dim"] - 1.0)

        initial_rover_positions[rov_id, 0] = rover_x
        initial_rover_positions[rov_id, 1] = rover_y
        initial_rover_positions[rov_id, 2] = rover_theta

    return initial_rover_positions


def rover_pos_center_concentrated():
    """
    Rovers given random starting positions within a radius of the center. Starting orientations are random.
    """
    radius = 8.0
    center_x = p["x_dim"]/2.0
    center_y = p["y_dim"]/2.0
    initial_rover_positions = np.zeros((p["n_rovers"], 3))

    for rov_id in range(p["n_rovers"]):
        x = random.uniform(0.0, p["x_dim"]-1.0)  # Rover X-Coordinate
        y = random.uniform(0.0, p["y_dim"]-1.0)  # Rover Y-Coordinate

        while x > (center_x + radius) or x < (center_x - radius):
            x = random.uniform(0.0, p["x_dim"]-1.0)  # Rover X-Coordinate
        while y > (center_y + radius) or y < (center_y - radius):
            y = random.uniform(0.0, p["y_dim"]-1.0)  # Rover Y-Coordinate

        initial_rover_positions[rov_id, 0] = x  # Rover X-Coordinate
        initial_rover_positions[rov_id, 1] = y  # Rover Y-Coordinate
        initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)  # Rover orientation

    return initial_rover_positions


def rover_pos_fixed_middle():  # Set rovers to fixed starting position
    """
    Rovers start out extremely close to the center of the map (they may be stacked).
    """
    initial_rover_positions = np.zeros((p["n_rovers"], 3))
    for rov_id in range(p["n_rovers"]):
        initial_rover_positions[rov_id, 0] = 0.5*p["x_dim"] + random.uniform(-1.0, 1.0)
        initial_rover_positions[rov_id, 1] = 0.5*p["y_dim"] + random.uniform(-1.0, 1.0)
        initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)

    return initial_rover_positions


# POI POSITION FUNCTIONS ------------------------------------------------------------------------------------------
def poi_pos_random(coupling):  # Randomly set POI on the map
    """
    POI positions set randomly across the map (but not too close to other POI).
    """
    pois_info = np.zeros((p["n_poi"], 4))  # [X, Y, Val, Coupling]

    for poi_id in range(p["n_poi"]):
        x = random.uniform(0, p["x_dim"]-1.0)
        y = random.uniform(0, p["y_dim"]-1.0)

        # Make sure POI don't start too close to one another
        poi_too_close = True
        while poi_too_close:
            count = 0
            for p_id in range(p["n_poi"]):
                if p_id != poi_id:
                    x_dist = x - pois_info[p_id, 0]
                    y_dist = y - pois_info[p_id, 1]

                    dist = math.sqrt((x_dist**2) + (y_dist**2))
                    if dist < (p["observation_radius"] - 1.5):
                        count += 1

            if count == 0:
                poi_too_close = False
            else:
                x = random.uniform(0, p["x_dim"] - 1.0)
                y = random.uniform(0, p["y_dim"] - 1.0)

        pois_info[poi_id, 0] = x
        pois_info[poi_id, 1] = y
        pois_info[poi_id, 3] = coupling

    return pois_info


def poi_pos_circle(coupling):
    """
    POI positions are set in a circle around the center of the map at a specified radius.
    """
    pois_info = np.zeros((p["n_poi"], 4))  # [X, Y, Val, Coupling]
    radius = 15.0
    interval = float(360/p["n_poi"])

    x = p["x_dim"]/2.0
    y = p["y_dim"]/2.0
    theta = 0.0

    for poi_id in range(p["n_poi"]):
        pois_info[poi_id, 0] = x + radius*math.cos(theta*math.pi/180)
        pois_info[poi_id, 1] = y + radius*math.sin(theta*math.pi/180)
        pois_info[poi_id, 3] = coupling
        theta += interval

    return pois_info


def poi_pos_concentric_circles(coupling):
    """
    POI positions are set in a circle around the center of the map at a specified radius.
    """
    assert(p["n_poi"] == 12)
    inner_radius = 6.5
    outter_radius = 15.0
    interval = float(360 / (p["n_poi"]/2))
    pois_info = np.zeros((p["n_poi"], 4))  # [X, Y, Val, Coupling]

    x = p["x_dim"]/2.0
    y = p["y_dim"]/2.0
    inner_theta = 0.0
    outter_theta = 0.0

    for poi_id in range(p["n_poi"]):
        if poi_id < 6:
            pois_info[poi_id, 0] = x + inner_radius * math.cos(inner_theta * math.pi / 180)
            pois_info[poi_id, 1] = y + inner_radius * math.sin(inner_theta * math.pi / 180)
            pois_info[poi_id, 3] = coupling
            inner_theta += interval
        else:
            pois_info[poi_id, 0] = x + outter_radius * math.cos(outter_theta * math.pi / 180)
            pois_info[poi_id, 1] = y + outter_radius * math.sin(outter_theta * math.pi / 180)
            pois_info[poi_id, 3] = coupling
            outter_theta += interval

    return pois_info


def poi_pos_two_poi(coupling):
    """
    Sets two POI on the map, one on the left, one on the right in line with global X-axis.
    """
    assert(p["n_poi"] == 2)
    pois_info = np.zeros((p["n_poi"], 4))  # [X, Y, Val, Coupling]

    # Left POI
    pois_info[0, 0] = 1.0
    pois_info[0, 1] = (p["y_dim"]/2.0) - 1
    pois_info[0, 3] = coupling

    # Right POI
    pois_info[1, 0] = p["x_dim"] - 2.0
    pois_info[1, 1] = (p["y_dim"]/2.0) + 1
    pois_info[1, 3] = coupling

    return pois_info


def poi_pos_four_corners(coupling):  # Statically set 4 POI (one in each corner)
    """
    Sets 4 POI on the map in a box formation around the center
    """
    assert(p["n_poi"] == 4)  # There must only be 4 POI for this initialization
    pois_info = np.zeros((p["n_poi"], 4))  # [X, Y, Val, Coupling]

    # Bottom left
    pois_info[0, 0] = 2.0
    pois_info[0, 1] = 2.0
    pois_info[0, 3] = coupling

    # Top left
    pois_info[1, 0] = 2.0
    pois_info[1, 1] = (p["y_dim"] - 2.0)
    pois_info[1, 3] = coupling

    # Bottom right
    pois_info[2, 0] = (p["x_dim"] - 2.0)
    pois_info[2, 1] = 2.0
    pois_info[2, 3] = coupling

    # Top right
    pois_info[3, 0] = (p["x_dim"] - 2.0)
    pois_info[3, 1] = (p["y_dim"] - 2.0)
    pois_info[3, 3] = coupling

    return pois_info


# POI VALUE FUNCTIONS -----------------------------------------------------------------------------------
def poi_vals_random(pois_info, v_low, v_high):
    """
    POI values randomly assigned 1-10
    """
    for poi_id in range(p["n_poi"]):
        pois_info[poi_id, 2] = float(random.randint(v_low, v_high))


def poi_vals_identical(pois_info, poi_val):
    """
    POI values set to fixed, identical value
    """
    for poi_id in range(p["n_poi"]):
        pois_info[poi_id, 2] = poi_val


def create_world_setup(coupling):
    """
    Create a new rover configuration file
    """

    # Initialize POI positions and values
    pois_info = np.zeros((p["n_poi"], 4))  # [X, Y, Val, Coupling]

    if p["poi_config_type"] == "Random":
        pois_info = poi_pos_random(coupling)
        poi_vals_random(pois_info, 3, 10)
    elif p["poi_config_type"] == "Two_POI":
        pois_info = poi_pos_two_poi(coupling)
        poi_vals_identical(pois_info, 10.0)
    elif p["poi_config_type"] == "Four_Corners":
        pois_info = poi_pos_four_corners(coupling)
        poi_vals_random(pois_info, 3.0, 10.0)
    elif p["poi_config_type"] == "Circle":
        pois_info = poi_pos_circle(coupling)
        poi_vals_random(pois_info, 3.0, 10.0)
    elif p["poi_config_type"] == "Con_Circle":
        pois_info = poi_pos_concentric_circles(coupling)
        poi_vals_random(pois_info, 3.0, 10.0)
    else:
        print("ERROR, WRONG POI CONFIG KEY")
    save_poi_configuration(pois_info)

    # Initialize Rover Positions
    initial_rover_positions = np.zeros((p["n_rovers"], 3))  # [X, Y, Theta]

    if p["rover_config_type"] == "Random":
        initial_rover_positions = rover_pos_random(pois_info)
    elif p["rover_config_type"] == "Concentrated":
        initial_rover_positions = rover_pos_center_concentrated()
    elif p["rover_config_type"] == "Fixed":
        initial_rover_positions = rover_pos_fixed_middle()

    save_rover_configuration(initial_rover_positions)


if __name__ == '__main__':
    """
    Create new world configuration files for POI and rovers
    """

    coupling = 1  # Default coupling requirement for POI
    rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))

    rd = RoverDomain()  # Number of POI, Number of Rovers
    create_world_setup(coupling)
    rd.load_world()

    for rover_id in range(p["n_rovers"]):
        for step in range(p["steps"]):
            rover_path[0:p["stat_runs"], rover_id, step, 0] = rd.rovers["R{0}".format(rover_id)].loc[0]
            rover_path[0:p["stat_runs"], rover_id, step, 1] = rd.rovers["R{0}".format(rover_id)].loc[1]
            rover_path[0:p["stat_runs"], rover_id, step, 2] = rd.rovers["R{0}".format(rover_id)].loc[2]

    create_pickle_file(rover_path, "Output_Data/", "Rover_Paths")
    run_visualizer()
