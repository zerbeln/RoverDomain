import math
import pickle
import csv
import os


def get_angle(target_x, target_y, xo, yo):
    """
    Calculates the angle between line connecting a target (target_x, target_y) and an origin point (x, y) with respect
    to the x-axis
    """
    dx = target_x - xo
    dy = target_y - yo

    angle = math.atan2(dy, dx) * (180.0 / math.pi)
    while angle < 0.0:
        angle += 360.0
    while angle > 360.0:
        angle -= 360.0
    if math.isnan(angle):
        angle = 0.0

    return angle


def get_squared_dist(target_x, target_y, xo, yo):
    """
    Calculates the squared distance between a target (target_x, target_y) and an origin point (x, y)
    """

    dx = target_x - xo
    dy = target_y - yo

    dist = (dx ** 2) + (dy ** 2)

    if dist < 1.0:
        dist = 1.0

    return dist


def get_linear_dist(target_x, target_y, xo, yo):
    """
    Calculates the linear distance between a target (target_x, target_y) and an origin point (x, y)
    """

    dx = target_x - xo
    dy = target_y - yo

    dist = math.sqrt((dx ** 2) + (dy ** 2))

    if dist < 1.0:
        dist = 1.0

    return dist


def create_csv_file(input_array, dir_name, file_name):
    """
    Save array as a CSV file in the specified directory
    """
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)
    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(input_array)


def create_pickle_file(input_data, dir_name, file_name):
    """
    Create a pickle file using provided data in the specified directory
    """

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    path_name = os.path.join(dir_name, file_name)
    rover_file = open(path_name, 'wb')
    pickle.dump(input_data, rover_file)
    rover_file.close()


def load_saved_policies(file_name, rover_id, srun):
    """
    Load saved Neural Network policies from pickle file
    """

    dir_name = 'Rover_Policies/Rover{0}/SRUN{1}'.format(rover_id, srun)
    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'rb')
    weights = pickle.load(weight_file)
    weight_file.close()

    return weights


def save_best_policies(network_weights, srun, file_name, rover_id):
    """
    Save trained neural networks for each rover as a pickle file
    """
    # Make sure Policy Bank Folder Exists
    if not os.path.exists('Rover_Policies'):  # If Data directory does not exist, create it
        os.makedirs('Rover_Policies')

    if not os.path.exists('Rover_Policies/Rover{0}'.format(rover_id)):
        os.makedirs('Rover_Policies/Rover{0}'.format(rover_id))

    dir_name = 'Rover_Policies/Rover{0}/SRUN{1}'.format(rover_id, srun)
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'wb')
    pickle.dump(network_weights, weight_file)
    weight_file.close()
