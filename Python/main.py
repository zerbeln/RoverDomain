from parameters import parameters as p
from rover_domain_example import rover_global, rover_difference, rover_dpp, test_trained_policies


if __name__ == '__main__':
    """
    Run classic rove domain using either G, D, or D++ for reward feedback.
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
    elif p["algorithm"] == "Test":
        print("Rover Domain: Testing Trained Policies")
        test_trained_policies()
    else:
        print("ALGORITHM TYPE ERROR")
