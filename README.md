# RoverDomain
Repository for core AADI rover domain. 

Core rover domain includes the following:
  Rover and POI class definitions in agent.py
  Rover Domain environment class (which includes global reward function) in rover_domain.py
  Definitions for Difference Rewards and D++ Rewards in reward_functions.py
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
