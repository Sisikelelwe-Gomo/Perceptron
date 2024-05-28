import numpy as np

# Dataset
data_points = np.array([
    [0.22, 0.33],
    [0.45, 0.76],
    [0.73, 0.39],
    [0.25, 0.35],
    [0.51, 0.69],
    [0.69, 0.42],
    [0.41, 0.49],
    [0.15, 0.29],
    [0.81, 0.32],
    [0.50, 0.88],
    [0.23, 0.31],
    [0.77, 0.30],
    [0.56, 0.75],
    [0.11, 0.38],
    [0.81, 0.33],
    [0.59, 0.77],
    [0.10, 0.89],
    [0.55, 0.09],
    [0.75, 0.35],
    [0.44, 0.55]
])

# Read cluster centers from standard input
centers = []
for _ in range(3):
    center_x = float(input())
    center_y = float(input())
    centers.append([center_x, center_y])
centers = np.array(centers)

# Initialize k
k = 3

# Step 4: Compute sum-of-squares error with respect to initial cluster centers
initial_error = np.sum(
    np.min(np.sum((data_points[:, np.newaxis, :] - centers) ** 2, axis=2), axis=1))

# Step 5: Perform one execution of the k-means algorithm
labels = np.argmin(
    np.sum((data_points[:, np.newaxis, :] - centers) ** 2, axis=2), axis=1)
new_centers = np.array([data_points[labels == i].mean(axis=0)
                       for i in range(k)])

# Step 6: Compute sum-of-squares error with respect to new cluster centers
new_error = np.sum(np.min(
    np.sum((data_points[:, np.newaxis, :] - new_centers) ** 2, axis=2), axis=1))

# Step 7: Output the sum-of-squares errors
print(round(initial_error, 4))
print(round(new_error, 4))
