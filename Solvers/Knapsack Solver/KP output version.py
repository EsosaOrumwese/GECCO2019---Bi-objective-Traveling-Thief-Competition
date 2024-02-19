import pandas as pd
import random
import numpy as np

class Item:
    def __init__(self, weight, profit, city_id):
        self.weight = weight
        self.profit = profit
        self.city_id = city_id
class Data:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.items = []
        self.process_data()

    def process_data(self):
        # Process the data from the CSV file
        for index, row in self.data.iterrows():
            weights = [int(x) for x in row['weights'].strip('[]').split()]
            profits = [int(x) for x in row['profit'].strip('[]').split()]
            city_id = row['city']
            for weight, profit in zip(weights, profits):
                self.items.append(Item(weight, profit, city_id))

        self.numItems = len(self.items)
        self.capacityOfKnapsack = 25936  # Adjust as needed
        print("Total number of items:", self.numItems)


def dp_with_random_profit(data_instance):
    numItems = data_instance.numItems
    cap = data_instance.capacityOfKnapsack
    
    # Create a copy of items with random profit fluctuation
    fluctuated_items = []
    for item in data_instance.items:
        random_profit = item.profit * (1 + random.uniform(-0.05, 0.05))  # +/- 5% fluctuation
        fluctuated_items.append(Item(item.weight, random_profit, item.city_id))
    
    tab = [[0] * (cap + 1) for _ in range(2)]
    items_in_solution = [0] * numItems

    for i in range(1, numItems + 1):
        item = fluctuated_items[i - 1]
        for j in range(1, cap + 1):
            if item.weight > j:
                tab[1][j] = tab[0][j]
            else:
                if tab[0][j - item.weight] + item.profit > tab[0][j]:
                    tab[1][j] = tab[0][j - item.weight] + item.profit
                    items_in_solution[i - 1] = 1
                else:
                    tab[1][j] = tab[0][j]
        tab[0], tab[1] = tab[1], tab[0]
    
        # Output the items included in the optimal solution
    return items_in_solution


    
    # Calculate total profit with original item profits for comparison
    #total_profit = sum(data_instance.items[i].profit for i in range(numItems) if items_in_solution[i])
    
    # Print the items included in the optimal solution and total profit
    #print(f'Total Profit with Original Profits: {total_profit}')  # Output total profit with original profits
    #print(f'Total Profit with Fluctuated Profits: {tab[0][cap]}')  # Output total profit with fluctuated profits
    #print('Items Selected:', items_in_solution)  # Output selected items
    #np.savetxt('TSP_solutions.txt', items_in_solution, fmt='%d')
    

# Use the modified function in the main logic
def main():
    instanceFileName = 'a.csv'  # Replace with your file name
    outputFileName = 'KP_Solutions.txt'
    
    # Generate multiple solutions with random profit fluctuations
    with open(outputFileName, 'w') as fout:
        for _ in range(12):
            data_instance = Data(instanceFileName)
            items_in_solution = dp_with_random_profit(data_instance)
            
            fout.write('[' + ','.join(map(str, items_in_solution)) + ']\n')
if __name__ == "__main__":
    main()
