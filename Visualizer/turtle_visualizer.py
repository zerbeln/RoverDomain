import turtle
import csv
import numpy as np
import pickle
import os
from parameters import parameters as p
import time


def import_rover_paths(config_id):
    """
    Import rover paths from pickle file
    """
    dir_name = 'Output_Data/'
    file_name = 'Rover_Paths{0}'.format(config_id)
    rover_path_file = os.path.join(dir_name, file_name)
    infile = open(rover_path_file, 'rb')
    rover_paths = pickle.load(infile)
    infile.close()

    return rover_paths


def import_poi_information(n_poi, config_id):
    """
    Import POI information from saved configuration files
    """
    pois = np.zeros((n_poi, 4))

    config_input = []
    with open('./World_Config/POI_Config{0}.csv'.format(config_id)) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            config_input.append(row)

    for poi_id in range(n_poi):
        pois[poi_id, 0] = float(config_input[poi_id][0])
        pois[poi_id, 1] = float(config_input[poi_id][1])
        pois[poi_id, 2] = float(config_input[poi_id][2])
        pois[poi_id, 3] = float(config_input[poi_id][3])

    return pois


def run_rover_visualizer(config_id):
    # Define screen parameters for the
    screen_width = p["x_dim"]*10
    screen_height = p["y_dim"]*10
    screen = turtle.Screen()
    screen.setup(screen_width+20, screen_height+20)  # define pixel width and height of screen
    screen.title("Rover Domain")
    screen.bgcolor("white")
    screen.tracer(0)

    rovers = []
    rover_paths = import_rover_paths(config_id)
    for rov_id in range(p["n_rovers"]):
        rovers.append(turtle.Turtle())
        rovers[rov_id].shape("circle")
        rovers[rov_id].shapesize(10 / 20)  # Number of pixels you want / 20 (default size)
        rovers[rov_id].color("blue")

    pois = []
    poi_info = import_poi_information(p["n_poi"], config_id)
    for poi_id in range(p["n_poi"]):
        pois.append(turtle.Turtle())
        pois[poi_id].shape("triangle")
        pois[poi_id].shapesize(20 / 20)  # Number of pixels you want / 20 (default size)
        pois[poi_id].color("red")
        pois[poi_id].penup()
        # Convert rover units to pixel units used by screen
        px = ((poi_info[poi_id, 0]/p["x_dim"]) * screen_width) - (screen_width/2)
        py = ((poi_info[poi_id, 1]/p["y_dim"]) * screen_height) - (screen_height/2)
        pois[poi_id].goto(px, py)
        pois[poi_id].stamp()

    for srun in range(p["stat_runs"]):
        for tstep in range(p["steps"]):
            for rov_id in range(p["n_rovers"]):
                rovers[rov_id].clearstamps()
                rovx = ((rover_paths[srun, rov_id, tstep, 0]/p["x_dim"])*screen_width) - (screen_width/2)
                rovy = ((rover_paths[srun, rov_id, tstep, 1]/p["y_dim"])*screen_height) - (screen_height/2)
                rovers[rov_id].goto(rovx, rovy)
                rovers[rov_id].stamp()
            screen.update()
            time.sleep(0.2)

    turtle.done()

