import numpy as np
import random
import copy


class Ccea:
    def __init__(self, p):
        self.population = {}
        self.fitness = np.zeros(p["pop_size"])
        self.pop_size = p["pop_size"]
        self.mut_rate = p["m_rate"]
        self.mut_chance = p["m_prob"]
        self.eps = p["epsilon"]
        self.fitness = np.zeros(self.pop_size)
        self.n_elites = p["n_elites"]  # Number of elites selected from each gen
        self.team_selection = np.ones(self.pop_size) * (-1)

        # Network Parameters that determine the number of weights to evolve
        self.n_inputs = p["n_inputs"]
        self.n_outputs = p["n_outputs"]
        self.n_hidden = p["n_hnodes"]

    def create_new_population(self):
        """
        Create new population of policies for rover NN
        :return: None
        """
        self.population = {}
        self.fitness = np.zeros(self.pop_size)
        self.team_selection = np.ones(self.pop_size) * (-1)

        for pop_id in range(self.pop_size):
            policy = {}
            policy["L1"] = np.random.rand(self.n_inputs * self.n_hidden)
            policy["L2"] = np.random.rand(self.n_hidden * self.n_outputs)
            policy["b1"] = np.random.rand(self.n_hidden)
            policy["b2"] = np.random.rand(self.n_outputs)

            self.population["pop{0}".format(pop_id)] = policy.copy()

    def select_policy_teams(self):
        """
        Choose the team number each policy in the population will be assigned to
        :return: None
        """
        self.team_selection = random.sample(range(self.pop_size), self.pop_size)

    def weight_mutate(self):
        """
        Mutate offspring population (each weight has a probability of being mutated)
        :return:
        """
        pop_id = int(self.n_elites)
        while pop_id < self.pop_size:
            mut_counter = 0
            # First Weight Layer
            for w in range(self.n_inputs*self.n_hidden):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    mut_counter += 1
                    weight = self.population["pop{0}".format(pop_id)]["L1"][w]
                    mutation = np.random.normal(0, self.mut_rate) * weight
                    self.population["pop{0}".format(pop_id)]["L1"][w] += mutation

            # Second Weight Layer
            for w in range(self.n_hidden*self.n_outputs):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    mut_counter += 1
                    weight = self.population["pop{0}".format(pop_id)]["L2"][w]
                    mutation = np.random.normal(0, self.mut_rate) * weight
                    self.population["pop{0}".format(pop_id)]["L2"][w] += mutation

            # Output bias weights
            for w in range(self.n_hidden):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    mut_counter += 1
                    weight = self.population["pop{0}".format(pop_id)]["b1"][w]
                    mutation = np.random.normal(0, self.mut_rate) * weight
                    self.population["pop{0}".format(pop_id)]["b1"][w] += mutation

            # Output layer weights
            for w in range(self.n_outputs):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    mut_counter += 1
                    weight = self.population["pop{0}".format(pop_id)]["b2"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop{0}".format(pop_id)]["b2"][w] += mutation

            pop_id += 1

    def epsilon_greedy_select(self):  # Choose K solutions
        """
        Select parents using e-greedy selection
        :return: None
        """
        new_population = {}
        for pop_id in range(self.pop_size):
            if pop_id < self.n_elites:
                new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(pop_id)])
            else:
                rnum = random.uniform(0, 1)
                if rnum < self.eps:
                    max_index = np.argmax(self.fitness)
                    new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(max_index)])
                else:
                    parent = random.randint(1, (self.pop_size - 1))
                    new_population["pop{0}".format(pop_id)] = copy.deepcopy(self.population["pop{0}".format(parent)])

        self.population = {}
        self.population = copy.deepcopy(new_population)

    def rank_population(self):
        """
        Reorders the population in terms of fitness (high to low)
        :return:
        """
        ranked_population = copy.deepcopy(self.population)
        for pop_id_a in range(self.pop_size):
            pop_id_b = pop_id_a + 1
            ranked_population["pop{0}".format(pop_id_a)] = copy.deepcopy(self.population["pop{0}".format(pop_id_a)])
            while pop_id_b < (self.pop_size):
                if pop_id_a != pop_id_b:
                    if self.fitness[pop_id_a] < self.fitness[pop_id_b]:
                        self.fitness[pop_id_a], self.fitness[pop_id_b] = self.fitness[pop_id_b], self.fitness[pop_id_a]
                        ranked_population["pop{0}".format(pop_id_a)] = copy.deepcopy(self.population["pop{0}".format(pop_id_b)])
                pop_id_b += 1

        self.population = {}
        self.population = copy.deepcopy(ranked_population)

    def down_select(self):  # Create a new offspring population using parents from top 50% of policies
        """
        Select parents create offspring population, and perform mutation operations
        :return: None
        """

        self.rank_population()
        self.epsilon_greedy_select()  # Select K successors using epsilon greedy
        self.weight_mutate()  # Mutate successors

    def reset_fitness(self):
        """
        Clears the fitness vector (helps catch errors but this function is not necessary)
        :return:
        """
        self.fitness = np.ones(self.pop_size)*-1
