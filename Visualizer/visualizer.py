import pygame
import numpy as np
import time
import math
import os
import csv
import pickle
from parameters import parameters as p

pygame.font.init()  # you have to call this at the start, if you want to use this module
myfont = pygame.font.SysFont('Comic Sans MS', 30)


def draw(display, obj, x, y):
    display.blit(obj, (x, y))  # Correct for center of mass shift


def generate_color_array(num_colors):  # Generates num random colors
    color_arr = []
    
    for i in range(num_colors):
        color_arr.append(list(np.array([255, 0, 0])))  # Red
        color_arr.append(list(np.array([0, 255, 0])))  # Green
        color_arr.append(list(np.array([0, 0, 255])))  # Blue
        color_arr.append(list(np.array([255, 0, 255])))  # Magenta
        color_arr.append(list(np.array([0, 255, 255])))  # Cyan
        color_arr.append(list(np.array([255, 255, 0])))  # Yellow
    
    return color_arr


def import_rover_paths(config_id):
    """
    Import rover paths from pickle file
    :return:
    """
    dir_name = 'Output_Data/'
    file_name = f'Rover_Paths{config_id}'
    rover_path_file = os.path.join(dir_name, file_name)
    infile = open(rover_path_file, 'rb')
    rover_paths = pickle.load(infile)
    infile.close()

    return rover_paths


def import_poi_information(n_poi, config_id):
    """
    Import POI information from saved configuration files
    :return:
    """
    pois = np.zeros((n_poi, 4))

    config_input = []
    with open(f'./World_Config/POI_Config{config_id}.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            config_input.append(row)

    for poi_id in range(n_poi):
        pois[poi_id, 0] = float(config_input[poi_id][0])
        pois[poi_id, 1] = float(config_input[poi_id][1])
        pois[poi_id, 2] = float(config_input[poi_id][2])
        pois[poi_id, 3] = float(config_input[poi_id][3])

    return pois


def run_visualizer(v_running=False, cf_id=0):
    """
    Run the visualizer that plots each rover's trajectory in the domain
    :return:
    """
    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    n_poi = p["n_poi"]
    rover_steps = p["steps"]
    obs_rad = p["observation_radius"]

    scale_factor = 10  # Scaling factor for images
    width = -15  # robot icon widths
    x_map = int(p["x_dim"] + 10)  # Slightly larger so POI are not cut off
    y_map = int(p["y_dim"] + 10)
    box_width = 1595  # 595 for 50x50, 1000 for 100x100
    rect = pygame.Rect((0, 0), (box_width, box_width))
    image_adjust = 100  # Adjusts the image so that everything is centered
    pygame.init()
    pygame.display.set_caption('Rover Domain')
    robot_image = pygame.image.load('Visualizer/rover.png')
    background = pygame.image.load('Visualizer/background.png')
    color_array = generate_color_array(17)
    pygame.font.init() 
    myfont = pygame.font.SysFont('Comic Sans MS', 30)

    rover_path = import_rover_paths(cf_id)
    pois = import_poi_information(n_poi, cf_id)

    poi_convergence = [0 for i in range(n_poi + 1)]
    for srun in range(stat_runs):
        game_display = pygame.display.set_mode((x_map * scale_factor, y_map * scale_factor))
        poi_status = [False for _ in range(n_poi)]
        for tstep in range(rover_steps):

            # Draw POI and calculate POI observations
            draw(game_display, background, 0, 0)
            pygame.draw.rect(game_display, (0, 0, 0), rect, 5)
            for poi_id in range(n_poi):  # Draw POI and POI values
                poi_x = int(pois[poi_id, 0] * scale_factor) + image_adjust
                poi_y = int(pois[poi_id, 1] * scale_factor) + image_adjust

                obs_count = 0
                for rover_id in range(n_rovers):
                    x_dist = pois[poi_id, 0] - rover_path[srun, rover_id, tstep, 0]
                    y_dist = pois[poi_id, 1] - rover_path[srun, rover_id, tstep, 1]
                    dist = math.sqrt((x_dist**2) + (y_dist**2))

                    if dist <= obs_rad:
                        obs_count += 1

                if obs_count >= pois[poi_id, 3]:
                    poi_status[poi_id] = True
                if poi_status[poi_id]:
                    pygame.draw.circle(game_display, (50, 205, 50), (poi_x, poi_y), 10)
                    pygame.draw.circle(game_display, (0, 0, 0), (poi_x, poi_y), int(obs_rad * scale_factor), 2)
                else:
                    pygame.draw.circle(game_display, (220, 20, 60), (poi_x, poi_y), 10)
                    pygame.draw.circle(game_display, (0, 0, 0), (poi_x, poi_y), int(obs_rad * scale_factor), 2)

                # This code displays the value of each PoI
                # textsurface = myfont.render(str(pois[poi_id, 2]), False, (0, 0, 0))
                # target_x = int(pois[poi_id, 0]*scale_factor) + image_adjust
                # target_y = int(pois[poi_id, 1]*scale_factor) + image_adjust
                # draw(game_display, textsurface, target_x, target_y)

            # Draw Rovers
            for rover_id in range(n_rovers):
                rover_x = int(rover_path[srun, rover_id, tstep, 0]*scale_factor) + width + image_adjust
                rover_y = int(rover_path[srun, rover_id, tstep, 1]*scale_factor) + width + image_adjust
                draw(game_display, robot_image, rover_x, rover_y)

                if tstep != 0:  # start drawing trails from timestep 1.
                    for timestep in range(1, tstep):  # draw the trajectory lines
                        line_color = tuple(color_array[rover_id])
                        start_x = int(rover_path[srun, rover_id, (timestep-1), 0]*scale_factor) + image_adjust
                        start_y = int(rover_path[srun, rover_id, (timestep-1), 1]*scale_factor) + image_adjust
                        end_x = int(rover_path[srun, rover_id, timestep, 0]*scale_factor) + image_adjust
                        end_y = int(rover_path[srun, rover_id, timestep, 1]*scale_factor) + image_adjust
                        line_width = 3
                        pygame.draw.line(game_display, line_color, (start_x, start_y), (end_x, end_y), line_width)
                        origin_x = int(rover_path[srun, rover_id, timestep, 0]*scale_factor) + image_adjust
                        origin_y = int(rover_path[srun, rover_id, timestep, 1]*scale_factor) + image_adjust
                        circle_rad = 3
                        pygame.draw.circle(game_display, line_color, (origin_x, origin_y), circle_rad)

            pygame.display.update()
            time.sleep(0.1)

        counter = 0
        for poi_id in range(n_poi):
            if poi_status[poi_id]:
                poi_convergence[poi_id] += 1
                counter += 1
        if counter == 0:
            poi_convergence[n_poi] += 1

        dir_name = './Screenshots/'  # Intended directory for output files
        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)
        image_name = "Screenshot_SR" + str(srun) + "_C" + str(cf_id) + ".jpg"
        screenshot_filename = os.path.join(dir_name, image_name)

        pygame.image.save(game_display, screenshot_filename)
        while v_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    v_running = False
