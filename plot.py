import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Ensure numpy is imported

csv_file_path = 'graph_data/lens.csv'


def plot_mtf_vs_perception(csv_file):
    data = pd.read_csv(csv_file)

    # Assuming the CSV has columns named 'MTF' and 'Perception'
    mtf = data['MTF']
    perception = data['Perception']

    # Plotting the graph with a smaller point size
    plt.figure(figsize=(10, 6))
    plt.plot(sorted(mtf), perception, marker='o', linestyle='-', color='b', markersize=4)  # Reduced point size here
    plt.title('MTF vs. Perception')
    plt.xlabel('MTF')
    plt.ylabel('Perception')

    # Setting the perception range from 0 to 1 with an interval of 0.1
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.grid(True)
    plt.show()


# Call the function with the path to your CSV file
plot_mtf_vs_perception(csv_file_path)
