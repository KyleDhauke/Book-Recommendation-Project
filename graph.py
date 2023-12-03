import matplotlib.pyplot as plt
from book_recommender_1 import *
import numpy as np


eigen_values = [1,2,3,4,6,9,14,21,32,50,77,119,183,282,436,673,1039,2475,3821]

var_value = [0.872,1.617, 2.315,2.988,4.163,5.578, 7.623,10.183,13.62,18.025,22.75,28.161,34.143,41.286,49.253, 57.951,67.52,89.685,100.0]

mean_times = [10.265,10.238,10.471,10.582,10.97,10.493,10.781,11.117,11.099,11.369,11.776,13.329,13.217, 14.415,16.626,20.808,26.944,69.142,123.4]
# Plot the eigen values vs. mean time taken
plt.plot(var_value, mean_times, marker='o', linestyle='-', color='blue')

# Add labels and title to the graph
plt.xlabel('Amount of Variance')
plt.ylabel('Mean Time Taken')
plt.title('Amount of Variance vs. Mean Time Taken')

# Show the plot
plt.show()



num_values = 20  # Number of values desired
min_eigen = 1  # Minimum eigen value
max_eigen = 3821  # Maximum eigen value



def eigen_values():
    eigen_values = np.logspace(np.log10(min_eigen), np.log10(max_eigen), num=num_values + 1).round()
    return eigen_values