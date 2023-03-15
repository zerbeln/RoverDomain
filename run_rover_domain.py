from parameters import parameters as p
from rover_training import rover_global, rover_difference, rover_dpp
from Visualizer.turtle_visualizer import run_rover_visualizer
from Visualizer.visualizer import run_visualizer


if __name__ == '__main__':
    """
    Run classic or tightly coupled rover domain using either G, D, D++, or CFL
    This main file is for use with rovers learning navigation (not skills)
    """

    if p["algorithm"] == "Global":
        print("Rover Domain: Global Rewards")
        rover_global()
    elif p["algorithm"] == "Difference":
        print("Rover Domain: Difference Rewards")
        rover_difference()
    elif p["algorithm"] == "DPP":
        print("Rover Domain: D++ Rewards")
        rover_dpp()
    else:
        print("ALGORITHM TYPE ERROR")
