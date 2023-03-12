import numpy as np
import random
import copy
from parameters import parameters as p


class EA:
    def __init__(self, n_inp=8, n_hid=10, n_out=2):
        self.population = {}
        self.pop_size = p["pop_size"]
        self.mut_rate = p["mutation_rate"]
        self.mut_chance = p["mutation_chance"]
        self.eps = p["epsilon"]
        self.fitness = np.zeros(self.pop_size)
        self.n_elites = p["n_elites"]  # Number of elites selected from each gen

        # Network Parameters that determine the number of weights to evolve
        self.n_inputs = n_inp
        self.n_outputs = n_out
        self.n_hidden = n_hid
        self.n_weights = ((n_inp+1) * n_hid) + ((n_hid + 1) * n_out)

    def reset_fitness(self):
        """
        Clear fitness vector of stored values
        """
        self.fitness = np.zeros(self.pop_size)

    def create_new_population(self):  # Re-initializes CCEA populations for new run
        """
        Create new populations (for beginning of stat run)
        """
        self.population = {}
        self.fitness = np.zeros(self.pop_size)

        for pol_id in range(self.pop_size):
            policy = {}
            policy['L1'] = np.random.normal(0, 1, self.n_inputs * self.n_hidden)
            policy['L2'] = np.random.normal(0, 1, self.n_hidden * self.n_outputs)
            policy['b1'] = np.random.normal(0, 1, self.n_hidden)
            policy['b2'] = np.random.normal(0, 1, self.n_outputs)

            self.population[f'pol{pol_id}'] = copy.deepcopy(policy)

    def weight_mutate(self):
        """
        Mutate offspring populations (each weight has a probability of mutation)
        """
        max_id = np.argmax(self.fitness)  # Preserve the champion

        for pol_id in range(self.pop_size):
            if pol_id != max_id:
                # First Weight Layer
                for w in range(self.n_inputs*self.n_hidden):
                    rnum1 = random.uniform(0, 1)
                    if rnum1 <= self.mut_chance:
                        weight = self.population[f'pol{pol_id}']['L1'][w]
                        weight += np.random.normal(0, self.mut_rate) * weight
                        self.population[f'pol{pol_id}']['L1'][w] = weight

                # Second Weight Layer
                for w in range(self.n_hidden*self.n_outputs):
                    rnum2 = random.uniform(0, 1)
                    if rnum2 <= self.mut_chance:
                        weight = self.population[f'pol{pol_id}']['L2'][w]
                        weight += np.random.normal(0, self.mut_rate) * weight
                        self.population[f'pol{pol_id}']['L2'][w] = weight

                # Output bias weights
                for w in range(self.n_hidden):
                    rnum3 = random.uniform(0, 1)
                    if rnum3 <= self.mut_chance:
                        weight = self.population[f'pol{pol_id}']['b1'][w]
                        weight += np.random.normal(0, self.mut_rate) * weight
                        self.population[f'pol{pol_id}']['b1'][w] = weight

                # Output layer weights
                for w in range(self.n_outputs):
                    rnum4 = random.uniform(0, 1)
                    if rnum4 <= self.mut_chance:
                        weight = self.population[f'pol{pol_id}']['b2'][w]
                        weight += (np.random.normal(0, self.mut_rate)) * weight
                        self.population[f'pol{pol_id}']['b2'][w] = weight

    def binary_tournament_selection(self):
        """
        Select parents using binary tournament selection
        """
        new_population = {}
        max_id = np.argmax(self.fitness)
        for pol_id in range(self.pop_size):
            if pol_id == max_id:  # Preserve the champion
                new_population[f'pol{pol_id}'] = copy.deepcopy(self.population[f'pol{max_id}'])
            else:
                p1 = random.randint(0, self.pop_size-1)
                p2 = random.randint(0, self.pop_size-1)
                while p1 == p2:
                    p2 = random.randint(0, self.pop_size - 1)

                if self.fitness[p1] > self.fitness[p2]:
                    new_population[f'pol{pol_id}'] = copy.deepcopy(self.population[f'pol{p1}'])
                elif self.fitness[p1] < self.fitness[p2]:
                    new_population[f'pol{pol_id}'] = copy.deepcopy(self.population[f'pol{p2}'])
                else:  # If fitnesses are equal, use a random tie breaker
                    rnum = random.uniform(0, 1)
                    if rnum > 0.5:
                        new_population[f'pol{pol_id}'] = copy.deepcopy(self.population[f'pol{p1}'])
                    else:
                        new_population[f'pol{pol_id}'] = copy.deepcopy(self.population[f'pol{p2}'])

        self.population = {}
        self.population = copy.deepcopy(new_population)

    def epsilon_greedy_select(self):  # Choose K solutions
        """
        Select parents using e-greedy selection
        """
        new_population = {}
        for pol_id in range(self.pop_size):
            if pol_id < self.n_elites:
                max_id = np.argmax(self.fitness)
                new_population[f'pol{pol_id}'] = copy.deepcopy(self.population[f'pol{max_id}'])
            else:
                rnum = random.uniform(0, 1)
                if rnum < self.eps:  # Greedy Selection
                    max_id = np.argmax(self.fitness)
                    new_population[f'pol{pol_id}'] = copy.deepcopy(self.population[f'pol{max_id}'])
                else:  # Random Selection
                    parent = random.randint(1, (self.pop_size - 1))
                    new_population[f'pol{pol_id}'] = copy.deepcopy(self.population[f'pol{parent}'])

        self.population = {}
        self.population = copy.deepcopy(new_population)

    def random_selection(self):
        """
        Choose next generation of policies using elite-random selection
        """
        new_population = {}
        for pol_id in range(self.pop_size):
            if pol_id < self.n_elites:
                max_id = np.argmax(self.fitness)
                new_population[f'pol{pol_id}'] = copy.deepcopy(self.population[f'pol{max_id}'])
            else:
                parent = random.randint(0, self.pop_size-1)
                new_population[f'pol{pol_id}'] = copy.deepcopy(self.population[f'pol{parent}'])

        self.population = {}
        self.population = copy.deepcopy(new_population)

    def rank_population(self):
        """
        Reorders the population in terms of fitness (high to low)
        """
        ranked_population = copy.deepcopy(self.population)
        for pol_a in range(self.pop_size):
            pol_b = pol_a + 1
            ranked_population[f'pol{pol_a}'] = copy.deepcopy(self.population[f'pol{pol_a}'])
            while pol_b < (self.pop_size):
                if pol_a != pol_b:
                    if self.fitness[pol_a] < self.fitness[pol_b]:
                        self.fitness[pol_a], self.fitness[pol_b] = self.fitness[pol_b], self.fitness[pol_a]
                        ranked_population[f'pol{pol_a}'] = copy.deepcopy(self.population[f'pol{pol_b}'])
                pol_b += 1

        self.population = {}
        self.population = copy.deepcopy(ranked_population)

    def down_select(self):  # Create a new offspring population using parents from top 50% of policies
        """
        Select parents create offspring population, and perform mutation operations
        """
        # self.rank_population()  # Sort populations in terms of fitness

        # Select K successors using desired selection strategy
        # self.epsilon_greedy_select()
        self.binary_tournament_selection()
        # self.random_selection()

        self.weight_mutate()  # Mutate successors
