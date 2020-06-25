import pygame
import numpy as np
import time
import math
import sys
import os
import csv
sys.path.append('../')

pygame.font.init()  # you have to call this at the start, if you want to use this module
myfont = pygame.font.SysFont('Comic Sans MS', 30)


def draw(display, obj, x, y):
    display.blit(obj, (x, y))  # Correct for center of mass shift


def generate_color_array(num_colors):  # Generates num random colors
    color_arr = []
    
    for i in range(num_colors):
        color_arr.append(list(np.random.choice(range(250), size=3)))
    
    return color_arr


def import_rover_paths(p):
    rover_paths = np.zeros((p["s_runs"], (p["n_steps"]+1), p["n_rovers"], 2))
    posFile = open('./Output_Data/Rover_Paths.txt', 'r')

    sr = 0
    rover_id = 0
    step_id = 0
    for line in posFile:
        if line == '\n':
            continue

        count = 0
        for coord in line.split('\t'):
            if (coord != '\n'):
                if count % 2 == 0:
                    rover_paths[sr, step_id, rover_id, 0] = float(coord)
                    count += 1
                else:
                    rover_paths[sr, step_id, rover_id, 1] = float(coord)
                    count += 1
                    step_id += 1
                if step_id == (p["n_steps"] + 1):
                    step_id = 0
        rover_id += 1
        if rover_id == p["n_rovers"]:
            rover_id = 0
            sr += 1

    return rover_paths


def import_poi_information(p):
    pois = np.zeros((p["n_poi"], 3))

    config_input = []
    with open('./Output_Data/POI_Config.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            config_input.append(row)

    for poi_id in range(p["n_poi"]):
        pois[poi_id, 0] = float(config_input[poi_id][0])
        pois[poi_id, 1] = float(config_input[poi_id][1])
        pois[poi_id, 2] = float(config_input[poi_id][2])

    return pois


def create_output_files(filename, in_vec):
    dir_name = 'Output_Data/'

    if not os.path.exists(dir_name):  # If directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, filename)
    with open(save_file_name, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(in_vec)


def run_visualizer(p):
    scale_factor = 20  # Scaling factor for images
    width = -15  # robot icon widths
    x_map = int(p["x_dim"] + 10)  # Slightly larger so POI are not cut off
    y_map = int(p["y_dim"] + 10)
    image_adjust = 100  # Adjusts the image so that everything is centered
    pygame.init()
    pygame.display.set_caption('Rover Domain')
    robot_image = pygame.image.load('./Visualizer/robot.png')
    background = pygame.image.load('./Visualizer/background.png')
    color_array = generate_color_array(p["n_rovers"])
    pygame.font.init() 
    myfont = pygame.font.SysFont('Comic Sans MS', 30)

    rover_path = import_rover_paths(p)
    pois = import_poi_information(p)

    poi_convergence = [0 for i in range(p["n_poi"] + 1)]
    for srun in range(p["s_runs"]):
        game_display = pygame.display.set_mode((x_map * scale_factor, y_map * scale_factor))
        poi_status = [False for _ in range(p["n_poi"])]
        for tstep in range(p["n_steps"]):
            draw(game_display, background, 0, 0)
            for poi_id in range(p["n_poi"]):  # Draw POI and POI values
                poi_x = int(pois[poi_id, 0] * scale_factor) + image_adjust
                poi_y = int(pois[poi_id, 1] * scale_factor) + image_adjust

                observer_count = 0
                for rover_id in range(p["n_rovers"]):
                    x_dist = pois[poi_id, 0] - rover_path[srun, tstep, rover_id, 0]
                    y_dist = pois[poi_id, 1] - rover_path[srun, tstep, rover_id, 1]
                    dist = math.sqrt((x_dist**2) + (y_dist**2))

                    if dist <= p["obs_rad"]:
                        observer_count += 1

                if observer_count >= p["c_req"]:
                    poi_status[poi_id] = True

                if poi_status[poi_id]:
                    pygame.draw.circle(game_display, (50, 205, 50), (poi_x, poi_y), 10)
                    pygame.draw.circle(game_display, (0, 0, 0), (poi_x, poi_y), int(p["obs_rad"] * scale_factor), 1)
                else:
                    pygame.draw.circle(game_display, (220, 20, 60), (poi_x, poi_y), 10)
                    pygame.draw.circle(game_display, (0, 0, 0), (poi_x, poi_y), int(p["obs_rad"] * scale_factor), 1)
                textsurface = myfont.render(str(pois[poi_id, 2]), False, (0, 0, 0))
                target_x = int(pois[poi_id, 0]*scale_factor) + image_adjust
                target_y = int(pois[poi_id, 1]*scale_factor) + image_adjust
                draw(game_display, textsurface, target_x, target_y)

            for rov_id in range(p["n_rovers"]):  # Draw all rovers and their trajectories
                rover_x = int(rover_path[srun, tstep, rov_id, 0]*scale_factor) + width + image_adjust
                rover_y = int(rover_path[srun, tstep, rov_id, 1]*scale_factor) + width + image_adjust
                draw(game_display, robot_image, rover_x, rover_y)

                if tstep != 0:  # start drawing trails from timestep 1.
                    for timestep in range(1, tstep):  # draw the trajectory lines
                        line_color = tuple(color_array[rov_id])
                        start_x = int(rover_path[srun, (timestep-1), rov_id, 0]*scale_factor) + image_adjust
                        start_y = int(rover_path[srun, (timestep-1), rov_id, 1]*scale_factor) + image_adjust
                        end_x = int(rover_path[srun, timestep, rov_id, 0]*scale_factor) + image_adjust
                        end_y = int(rover_path[srun, timestep, rov_id, 1]*scale_factor) + image_adjust
                        line_width = 3
                        pygame.draw.line(game_display, line_color, (start_x, start_y), (end_x, end_y), line_width)
                        origin_x = int(rover_path[srun, timestep, rov_id, 0]*scale_factor) + image_adjust
                        origin_y = int(rover_path[srun, timestep, rov_id, 1]*scale_factor) + image_adjust
                        circle_rad = 3
                        pygame.draw.circle(game_display, line_color, (origin_x, origin_y), circle_rad)

            pygame.display.update()
            time.sleep(0.1)

        counter = 0
        for poi_id in range(p["n_poi"]):
            if poi_status[poi_id]:
                poi_convergence[poi_id] += 1
                counter += 1
        if counter == 0:
            poi_convergence[p["n_poi"]] += 1

        dir_name = 'Screenshots/'  # Intended directory for output files
        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)
        image_name = "Screenshot_SR" + str(srun) + ".jpg"
        screenshot_filename = os.path.join(dir_name, image_name)

        pygame.image.save(game_display, screenshot_filename)
        while p["running"]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    p["running"] = False

    create_output_files('POI_Choice.csv', poi_convergence)
