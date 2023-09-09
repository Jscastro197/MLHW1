import numpy as np
import pandas as pd

data_csv = pd.read_csv("telescope_data.csv")

# Identify non-numeric columns
non_numeric_columns = data_csv.select_dtypes(exclude=[np.number]).columns.tolist()

# Convert non-numeric columns to numeric
data_csv[non_numeric_columns] = data_csv[non_numeric_columns].apply(pd.to_numeric, errors='coerce')

data_csv.iloc[:, :-1].values

# Convert the DataFrame to a numpy array
data = data_csv.to_numpy()

# Calculate the multivariate mean vector
mean_vector = np.mean(data, axis=0)
print(mean_vector)

print("2. Compute the sample covariance matrix as inner products between the columns of the centered data matrix")
centered_data = data - mean_vector

cov1 = np.dot(centered_data.T,centered_data)/centered_data.shape[0]

print(cov1)