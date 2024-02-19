# Optimization of the Bi-objective Travelling Thief Problem for the GECCO2019 Competition

## Introduction

This repository contains an attempt, although yet not perfected, at the Python implementation of a non-dominated sorting biased random-key genetic algorithm (NDSBRKGA) by Chagas et al [1] to solve the problem given at GECCO2019 - Bi-objective Traveling Thief Competition. 

The goal of this competition was to provide a platform for researchers in computational intelligence working on multi-component optimization problems. The main focus of this competition was on the combination of Travelling Salesman Problem (TSP) and Knapsack Problems (KP). The TSP problem one in which the salesman has $n$ cities and the task is to find the best possible route which minimizes the total distance. While for the KP problem, a knapsack has to be filled with items of value $b_j$ and weight $w_j$ without violating the maximum weight constraint $Q$ while maximizing the total profit.

For this Travelling Thief Problem, the two objectives become interwoven because the thief has to go through $n$ cities each with $m$ items and fill up the knapsack without violating the $Q$. And as a result of the weight of items in his knapsack, his velocity $v$ reduces and so does the time of the tour. So the thief’s two objectives now include minimizing the total tour time while maximizing
the profit gained. So the weight of items is both considered in the KP aspect of the problem, as $Q$ can’t be violated, as well as in the velocity of the thief as he travels between cities in his tour.

## Requirements

- Python 3.x
- Required Python packages: [`numpy`, `pickle`, `ast`]

## Usage

1. Install the required packages:

2. Run the GA solver with the desired TTP instance file.

distance_fl = [`distance_matrix`, `distance_matrix_a280-n2790.pkl`, `distance_matrix_a280-n2790.pkl`, `distance_matrix_a280-n2790.pkl`, `distance_matrix_fnl4461-n4460.pkl`, `distance_matrix_fnl4461-n4460.pkl`, `distance_matrix_fnl4461-n4460.pkl`]

TSP_fl = [`TSP_solutions`, `TSP_solutions_a280-n2790.txt`, `TSP_solutions_a280-n2790.txt`, `TSP_solutions_a280-n2790.txt`, `TSP_solutions_fnl4461-n4460.txt`, `TSP_solutions_fnl4461-n4460.txt`, `TSP_solutions_fnl4461-n4460.txt`]

KP_fl = [`KP_Solutions`, `a280-n279 KP_Solutions.txt`, `a280-n1395 KP_Solutions.txt`,`a280-n2790 KP_Solutions.txt`, `fnl4461-n4460 KP_Solutions.txt`, `fnl4461-n22300 KP_Solutions.txt`, `fnl4461-n44600 KP_Solutions.txt`]


## Input File Format

There are six datasets can be tested.
File_names = [`a.csv`, `a280-n279.csv` , `a280-n1395.csv`, `a280-n2790.csv`, `fnl4461-n4460.csv`, `fnl4461-n22300.csv`, `fnl4461-n44600.csv`]

However, due to the long computation time, the algorithm was tested on the smallest dataset i.e. `a280-n279.csv` and this still took ~3 days. 

## Components of the solver
1. Population Initialization:
Generating an initial population based on TSP and KP solutions, which is a crucial step for starting the genetic algorithm.
2. Fitness Function:
The get_fitness function evaluates the quality of each solution. It considers both the TSP (time to traverse the route) and KP (total profit from selected items) aspects, providing a multi-objective evaluation.
3. Encoding and Decoding:
These processes translate between the phenotype (actual solution representation) and genotype (genetic representation). This translation is vital for applying genetic operations while keeping the solutions interpretable.
4. Repair Operator:
Ensures that solutions adhere to the KP constraints, particularly the capacity limit. This step maintains the feasibility of the solutions throughout the genetic process.
5. NSGA-II for Non-dominated Sorting:
This technique is used to identify Pareto-optimal fronts among the solutions, an essential part of multi-objective optimization.
6. Local Search Integration:
Incorporates 2-opt (for TSP) and bit-flip (for KP) to improve the quality of solutions. This step is crucial for exploring the solution space more thoroughly.
7. Biased Crossover and Mutation:
These genetic operators introduce variability and new traits into the population, crucial for exploring diverse solutions and avoiding local optima.
8. Termination and Result Compilation:
The process iterates until a set number of epochs, ensuring sufficient exploration of the solution space. The final results are compiled and analyzed for both objectives (TSP and KP).

## Analysis and Discussion
Multi-Objective Optimization: This approach effectively combines TSP and KP, two different optimization problems, into a single framework, addressing the complexities inherent in multi-objective optimization.

Efficiency and Scalability: The solver seems well-suited for large datasets and complex problem spaces, although the computational intensity might be high given the multiple layers of processing and evaluation.

Robustness and Solution Quality: The use of NSGA-II, local search techniques, and genetic operators enhances the robustness of the approach, likely leading to high-quality solutions.

Visualization and Analysis: The inclusion of visualization (scatter plot of fitness values and TSP fitness trend) is to analyze the effectiveness and progression of the algorithm over iterations.

## Output

The solver will provide a dictionary containing the best solutions found during the optimization process. The dictionary structure is as follows:

Feel free to explore and modify the code to suit your specific needs.

## Contribution
The contributers of this project include;
* Bhagat, Kanav
* Xiao, Xiao
* Orumwese, Esosa
* Xia, Junjie


## Acknowledgments
This implementation is based on the principles of Genetic Algorithms and their application to combinatorial optimization problems. If you find this code useful, consider referencing the relevant literature on TTP and GA.

[1.] J. B. Chagas, J. Blank, M. Wagner, M. J. Souza, and K. Deb, “A non-dominated sorting based customized random-key genetic algorithm for the bi-objective traveling thief problem,” Journal of Heuristics, vol. 27, no. 3, pp. 267–301, 2021.

