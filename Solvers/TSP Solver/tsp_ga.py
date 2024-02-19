from sys import maxsize
from time import time
from random import random, randint, sample

import numpy as np
import pandas as pd
from haversine import haversine


class Gene:  # City
    # keep distances from cities saved in a table to improve execution time.
    __distances_table = {}

    def __init__(self, name, lat, lng):
        self.name = name
        self.lat = lat
        self.lng = lng

    def get_distance_to(self, dest):
        origin = (self.lat, self.lng)

        forward_key = (self.name, dest.name)
        backward_key = (dest.name, self.name)

        # Get distance
        if forward_key in Gene.__distances_table:
            return Gene.__distances_table[forward_key]

        if backward_key in Gene.__distances_table:
            return Gene.__distances_table[backward_key]

        x1, y1 = origin[0], origin[1]
        x2, y2 = dest.lat, dest.lng

        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        Gene.__distances_table[forward_key] = dist
        return dist


class Individual():  # Route: possible solution to TSP
    def __init__(self, genes, start, distances):
        assert(len(genes) > 3)
        self.genes = genes
        self.start = start
        self.__reset_params()
        self.distance = distances


    def swap(self, gene_1, gene_2):
        self.genes[0]
        a, b = self.genes.index(gene_1), self.genes.index(gene_2)
        self.genes[b], self.genes[a] = self.genes[a], self.genes[b]
        self.__reset_params()

    def add(self, gene):
        self.genes.append(gene)
        self.__reset_params()

    @property
    def fitness(self):
        if self.__fitness == 0:
            self.__fitness = 1 / self.travel_cost  # Normalize travel cost
        return self.__fitness

    @property
    def travel_cost(self):  # Get total travelling cost
        if self.__travel_cost == 0:
            for i in range(len(self.genes)):
                origin = self.genes[i]
                if i == len(self.genes) - 1:
                    dest = self.genes[0]
                else:
                    dest = self.genes[i+1]

                #self.__travel_cost += origin.get_distance_to(dest)
                self.__travel_cost += self.distance[int(origin.name), int(dest.name)]
            #self.__travel_cost += self.start.get_distance_to(self.genes[-1])
            self.__travel_cost += self.distance[int(self.start.name), int(self.genes[-1].name)]

        return self.__travel_cost

    def __reset_params(self):
        self.__travel_cost = 0
        self.__fitness = 0


class Population:  # Population of individuals
    def __init__(self, individuals, start):
        self.individuals = individuals
        self.start = start

    @staticmethod
    def gen_individuals(sz, genes, city0, distance_m):
        individuals = []
        for _ in range(sz):
            individuals.append(Individual(sample(genes, len(genes)), city0, distances=distance_m))
        return Population(individuals,city0)

    def add(self, route):
        self.individuals.append(route)

    def rmv(self, route):
        self.individuals.remove(route)

    def get_fittest(self):
        fittest = self.individuals[0]
        for route in self.individuals:
            if route.fitness > fittest.fitness:
                fittest = route
        return fittest


def evolve(pop, tourn_size, mut_rate, start, distance_matrix):
    new_generation = Population([], start)
    pop_size = len(pop.individuals)
    elitism_num = pop_size // 2

    # Elitism
    for _ in range(elitism_num):
        fittest = pop.get_fittest()
        new_generation.add(fittest)
        pop.rmv(fittest)

    # Crossover
    for _ in range(elitism_num, pop_size):
        parent_1 = selection(new_generation, tourn_size, start)
        parent_2 = selection(new_generation, tourn_size, start)
        child = crossover(parent_1, parent_2, start, distance_m=distance_matrix)
        new_generation.add(child)

    # Mutation
    for i in range(elitism_num, pop_size):
        mutate(new_generation.individuals[i], mut_rate)

    return new_generation


def crossover(parent_1, parent_2, city0, distance_m):
    # Partially Mapped Crossover
    def fill_with_parent1_genes(child, parent, genes_n):
        start_at = randint(0, len(parent.genes)-genes_n-1)
        finish_at = start_at + genes_n
        for i in range(start_at, finish_at):
            child.genes[i] = parent_1.genes[i]

    def fill_with_parent2_genes(child, parent):
        j = 0
        for i in range(0, len(parent.genes)):
            if child.genes[i] == None:
                while parent.genes[j] in child.genes:
                    j += 1
                child.genes[i] = parent.genes[j]
                j += 1

    genes_n = len(parent_1.genes)
    child = Individual([None for _ in range(genes_n)], city0, distances=distance_m)
    fill_with_parent1_genes(child, parent_1, genes_n // 2)
    fill_with_parent2_genes(child, parent_2)

    return child


def mutate(individual, rate):
    # swap mutation
    for _ in range(len(individual.genes)):
        if random() < rate:
            sel_genes = sample(individual.genes, 2)
            individual.swap(sel_genes[0], sel_genes[1])


def selection(population, competitors_n, city0):
    return Population(sample(population.individuals, competitors_n), city0).get_fittest()



def run_ga(genes, pop_size, n_gen, tourn_size, mut_rate, verbose, distances):
    # Get start, and make remained into the populations.
    start = genes[0]
    remain = genes[1:]
    # Initial Solutions
    population = Population.gen_individuals(pop_size, remain, start, distance_m=distances)
    # 'cost' to store historical cost
    history = {'cost': [population.get_fittest().travel_cost], 'good_routes': []}
    counter, generations, min_cost = 0, 0, maxsize

    if verbose:
        print("-- TSP-GA -- Initiating evolution...")
    start_time = time()
    # Number of generations
    while counter < n_gen:
        # Generate Result
        population = evolve(population, tourn_size, mut_rate, start, distance_matrix=distances)
        cost = population.get_fittest().travel_cost
        if cost < min_cost:
            counter, min_cost = 0, cost
        else:
            counter += 1

        generations += 1
        route = [start.name]+[ge.name for ge in population.get_fittest().genes]
        if not route in history['good_routes']:
            history['good_routes'].append(route)
        history['cost'].append(cost)
    total_time = round(time() - start_time, 6)

    if verbose:
        print("-- TSP-GA -- Evolution finished after {} generations in {} s".format(generations, total_time))
        print("-- TSP-GA -- Minimum travelling cost {} KM".format(min_cost))

    history['generations'] = generations
    history['total_time'] = total_time
    history['route'] = history['good_routes'][-1]
    history['best_cost'] = min_cost
    history['good_routes'] = history['good_routes'][-20:] if len(history['good_routes']) > 20 else history['good_routes']

    return history