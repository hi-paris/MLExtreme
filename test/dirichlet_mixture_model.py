import matplotlib.pyplot as plt
import numpy as np
import MLExtreme as mlx

# temporary, working version, to be removed
import os
os.getcwd()
os.chdir("../")
os.chdir("MLExtreme")
# import importlib
# importlib.reload(mlx)
# from MLExtreme.utils import dataset_generation as dg

# test for function rdirichlet

# Parameters for the test
n = 100  # Number of samples
p = 2    # Number of components in the Dirichlet distribution

# Define two constant alpha vectors for the two groups
alpha1 = np.array([10, 20])  # First constant alpha for the first n/2 rows
alpha2 = np.array([30, 10])  # Second constant alpha for the last n/2 rows

# Create the alpha matrix by repeating the two alpha vectors
alpha = np.vstack([np.tile(alpha1, (n//2, 1)), np.tile(alpha2, (n//2, 1))])

# Generate the Dirichlet samples
samples = mlx.gen_dirichlet(n, alpha)


# Plot the generated data as a scatter plot with transparency (grey points)
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], color='grey', alpha=0.5,
            label='Generated Samples')

# Labels and title
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Scatter plot of Dirichlet Samples with Two Different Alphas')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# test for function gen_dirimix (generating angular datasets from
# a dirichlet mixture)
p = 3  # Number of components in each Dirichlet distribution
k = 4  # Number of mixture components
n = 10  # Number of samples

Mu = np.random.rand(p, k)  # p x k matrix of means
lnu = np.random.rand(k)  # k vector of log-scales
wei = np.random.rand(k)  # k vector of weights

# Generate the mixture samples
samples = mlx.gen_dirimix(n, Mu, lnu, wei)

# Print the generated samples
print(samples)

# plot example of dirichlet mixture


# Parameters for the example
n = 500  # Number of samples
p = 2    # Number of components in the Dirichlet distribution
k = 2    # Number of mixture components

# Mixture means (Mu), log scale (lnu), and weights (wei)
Mu = np.array([[0.2, 0.7], [0.8, 0.3]])  # p x k matrix of means
lnu = np.log(20) * np.ones(k)  # log(10) for both components
wei = np.array([0.2, 0.8])  # weights for the mixture components

# Generate samples using the mixture of Dirichlet distributions
samples = mlx.gen_dirimix(n, Mu, lnu, wei)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], color='grey', alpha=0.5)
plt.title(
    "Samples from a Mixture of Dirichlet Distributions (p=2, n=500, k=2)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()

# density plots  with angular density plotting functions #

#  (2D example)
Mu = np.array([[0.2, 0.7], [0.8, 0.3]])  # Means of the two components
lnu = np.log(10) * np.ones(2)  # log(10) for each component
wei = np.array([0.1, 0.9])  # Mixture weights

# Plot the mixture density

mlx.plot_pdf_dirimix_2D(Mu, lnu, wei)

# 3D example (angular density)
Mu = np.array([[0.2, 0.7], [0.8, 0.3], [0.5, 0.5]])  # Means for 3 components
lnu = np.log(10) * np.ones(2)  # log(10) for each component
wei = np.array([0.2, 0.8])  # Mixture weights

# Plot the mixture density on the 2D simplex

mlx.plot_pdf_dirimix_3D(Mu, lnu, wei)

# Examples of dataset generation
# in the tail model for regular variation
# X = R * Theta
# R is Pareto ; Theta ~ Dirichlet mixture
# This type of sdata is called RV-dirimix from now on

# Example RV-dirimix  data generation p = 2
n = 5000  # Number of samples
p = 2  # Dimensionality of the simplex (2D in this case)
k = 3  # Number of components in the Dirichlet mixture
alpha = 2.5  # Shape parameter of the Pareto distribution


# Parameters for the Dirichlet mixture
Mu = np.array([[0.3, 0.5,  0.7], [0.7, 0.5,  0.3]])
# Means for the mixture (2D)
wei = np.array([0.3, 0.4, 0.3])  # Weights for each component
lnu = np.log(np.array([20, 7, 4]))
# Log scale parameters for the mixture components
# inspect the angular density
Mu_wei = Mu @ wei  # Equivalent to np.dot(Mu, wei)
# Display the result
print(Mu_wei)
mlx.plot_pdf_dirimix_2D(Mu, lnu, wei)

# Generate the dataset using the gen_rv_dirimix function
X = mlx.gen_rv_dirimix(alpha, Mu, wei, lnu, index_weight_noise=0.5**alpha, size=n)

# Plotting the dataset (2D plot)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], color='grey', alpha=0.5)
plt.xlabel('X1')
plt.ylabel('X2')

# Calculate the equal axis limit based on the maximum range
max_range = max(np.max(X[:, 0]), np.max(X[:, 1]))

# Set both axes to have the same scale
plt.xlim(0, max_range)
plt.ylim(0, max_range)

plt.title('2D Scatter Plot of Generated Points')
plt.show()

# Example usage p = 3
n = 500  # Number of samples
p = 3  # Dimensionality of the simplex
k = 2  # Number of components in the Dirichlet mixture
alpha = 2.5  # Shape parameter of the Pareto distribution

# Parameters for the Dirichlet mixture
Mu = np.array([[0.2, 0.7], [0.8, 0.3], [0.5, 0.5]])  # Means for the mixture
wei = np.array([0.2, 0.8])  # Weights for each component
lnu = np.log(np.array([10, 5]))  # Log scale parameters

# Generate the dataset
X = mlx.gen_rv_dirimix(alpha, Mu, wei, lnu, size=n)

# from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='gray', alpha=0.5)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()
