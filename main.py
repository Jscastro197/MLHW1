import numpy as np
import csv

with open('telescope_data.csv', 'r') as f:
  reader = csv.reader(f)
  data_csv = list(reader)

#   print(data_csv)

# Create your data matrix (replace this with your actual data)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Compute the mean of each column
mean_vector = np.mean(data, axis=0)
print(mean_vector)

# Center the data by subtracting the mean from each column
centered_data = data - mean_vector

# Compute the sample covariance matrix
covariance_matrix = np.cov(centered_data, rowvar=False)

# rowvar=False means that variables are represented by columns (not rows)

# print("Sample Covariance Matrix:")