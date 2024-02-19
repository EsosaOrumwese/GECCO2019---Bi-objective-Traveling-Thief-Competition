import time

import numpy as np
import pandas as pd

import utils
import random
import argparse
import tsp_ga as ga
from datetime import datetime

def calculate_distance_matrix(city_names, x_coordinates, y_coordinates):
    num_cities = len(city_names)
    distance_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            # Calculate Euclidean distance between cities i and j using coordinates
            distance = np.sqrt((x_coordinates[i] - x_coordinates[j])**2 + (y_coordinates[i] - y_coordinates[j])**2)
            distance_matrix[i][j] = distance
    return distance_matrix

def run(args):
    # document path
    genes = utils.get_genes_from(args.cities_fn)
    # Number of cities
    if args.verbose:
        print("-- Running TSP-GA with {} cities --".format(len(genes)))
    whole_df = pd.read_csv(args.cities_fn)
    distance_matrix = calculate_distance_matrix(whole_df['city'], whole_df['longitude'], whole_df['latitude'])
    history = ga.run_ga(genes, args.pop_size, args.n_gen,
                        args.tourn_size, args.mut_rate, args.verbose, distances=distance_matrix)
    history['distance_matrix'] = distance_matrix
    print('generations', history['generations'])
    print('total_time', history['total_time'])

    print('distance_matrix', distance_matrix)
    print('cost', history['cost'][-1])
    print('good_routes', history['good_routes'])

    if args.verbose:
        print("-- Drawing Route --")

    utils.plot(history['cost'], history['route'])

    if args.verbose:
        print("-- Done --")

    return history


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    # Read the TSP part of data
    df = pd.read_csv('a.csv')
    cities = df[:21][['city', 'X', 'Y']]
    new_cities = cities.rename(columns={'X': 'longitude', 'Y': 'latitude'})
    new_cities.to_csv('cities.csv', index=True)

    # Set Parameters
    parser.add_argument('-v', '--verbose', type=int, default=1)
    parser.add_argument('--pop_size', type=int, default=500, help='Population size')
    parser.add_argument('--tourn_size', type=int, default=50, help='Tournament size')
    parser.add_argument('--mut_rate', type=float, default=0.02, help='Mutation rate')
    parser.add_argument('--n_gen', type=int, default=20, help='Number of equal generations before stopping')
    parser.add_argument('--cities_fn', type=str, default="cities.csv", help='Data containing the geographical coordinates of cities')

    random.seed(666)
    args = parser.parse_args()
    if args.tourn_size > args.pop_size:
        raise argparse.ArgumentTypeError('Tournament size cannot be bigger than population size.')

    # result.(generations,total_time,route,distance_matrix)
    result = run(args)

    TSP_solutions = result['good_routes']
    distances = result['distance_matrix']

    np.savetxt('TSP_solutions.txt', TSP_solutions, fmt='%d')
    import pickle
    with open('distance_matrix.pkl','wb') as file:
        pickle.dump(distances, file)

    end_time = time.time()
    run_time = round(end_time - start_time)
    hour = run_time // 3600
    minute = (run_time - hour * 3600) // 60
    second = run_time - 3600 * hour - 60 * minute
    print(f'Total Run time：{hour}hours{minute}minutes{second}seconds')