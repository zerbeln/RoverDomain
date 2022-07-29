# RoverDomain
Repository for core AADI rover domain. The rover domain runs as an AI gym style environment. The "step" function in the rover domain takes in rover actions from the neural networks and returns the next state and the global reward for that state. Rover actions are formatted as an array that is (n x a) where n is the number of rovers and a is the number of action outputs from the network. The global reward output at each step is an array showing the reward returned by each POI. This array is (p x 1) where p is the number of POI in the environment.

Core rover domain includes the following:
  Rover and POI class definitions in agent.py
  Rover Domain environment class (which includes global reward function) in rover_domain.py
  Definitions for Difference Rewards and D++ Rewards in reward_functions.py
  A basic fully connected, feedforward neural network for rovers in rover_neural_network.py
  A basic CCEA with several selection strategies in ccea.py
  Typical parameters used for test setup in parameters.py
  Supporting functions in global_functions.py
  A script that creates a rover domain environment with specified numbers of POI and Roves in create_world.py
  An example of how to setup the rover domain and run it to collect data in rover_domain_example.py.
  
  To get started:
  1. Set your parameters in parameters.py.
  2. Run python3 create_world.py to setup the environment. This will save the environment configuration in a directory called World_Config.
  3. Run python3 main.py to execute your test.
  4. Some basic performance data will be saved to a directory called Output_Data.
  
  To use this codebase you will need the following for the Python implementation:
  1. A functioning Python 3 environment (3.6 or higher)
  2. Numpy
  3. Pygame for the visualizer
