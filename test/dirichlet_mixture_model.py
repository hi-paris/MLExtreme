import matplotlib.pyplot as plt
import numpy as np
import MLExtreme as mlx

# temporary, working version, to be removed
import os
os.getcwd()
# os.chdir("../")

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
samples = mlx.gen_dirichlet(alpha, size=n)


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

Mu = np.random.rand(k, p)# k* p  matrix of means
lnu = np.random.rand(k)  # k vector of log-scales
wei = np.random.rand(k)  # k vector of weights

# Generate the mixture samples
samples = mlx.gen_dirimix(Mu, wei, lnu, size=n)

# Print the generated samples
print(samples)

# plot example of dirichlet mixture


# Parameters for the example
n = 500  # Number of samples
p = 2    # Number of components in the Dirichlet distribution
k = 3    # Number of mixture components

# Mixture means (Mu), log scale (lnu), and weights (wei)
Mu = np.array([[0.2, 0.8], [0.8, 0.2], [0.5,0.5]])  # k* p matrix of means
lnu = np.log(20) * np.ones(k)  # log(10) for both components
wei = np.array([0.5, 0.2, 0.3])  # weights for the mixture components

# normalize the parameters so that the generated random vector h as
# unit pareto margins when the Dirichlet angle is multiplied by a
# pareto(1) radius

Mu, wei = mlx.normalize_param_dirimix(Mu, wei)

# Generate samples using the mixture of Dirichlet distributions
samples = mlx.gen_dirimix(Mu, wei, lnu, size=n)

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

# Plot the mixture density

mlx.plot_pdf_dirimix_2D(Mu,  wei, lnu)

# 3D example (angular density)
Mu = np.array([[0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])  # Means for 3 components
lnu = np.log(15) * np.ones(2)  # log(10) for each component
wei = np.array([0.5, 0.5])  # Mixture weights
Mu,wei = mlx.normalize_param_dirimix(Mu,wei)

# Plot the mixture density on the 2D simplex

mlx.plot_pdf_dirimix_3D(Mu, wei, lnu)

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

# Mixture means (Mu), log scale (lnu), and weights (wei)
Mu = np.array([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5]])  # k* p matrix of means
lnu = np.log(20) * np.ones(k)  # log(10) for both components
wei = np.array([0.5, 0.2, 0.3])  # weights for the mixture components
Mu, wei = mlx.normalize_param_dirimix(Mu, wei)
# inspect the angular density
Mu_wei = wei  @ Mu
# Display the result
print(Mu_wei)
mlx.plot_pdf_dirimix_2D(Mu, wei, lnu)

# Generate the dataset using the gen_rv_dirimix function
X = mlx.gen_rv_dirimix(alpha, Mu, wei, lnu, index_weight_noise=0.5**alpha,
                       size=n)

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
Mu = np.array([[0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])  # Means for 3 components
lnu = np.log(15) * np.ones(2)  # log(10) for each component
wei = np.array([0.5, 0.5])  # Mixture weights
Mu,wei = mlx.normalize_param_dirimix(Mu,wei)

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
