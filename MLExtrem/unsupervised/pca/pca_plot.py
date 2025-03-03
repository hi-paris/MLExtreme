import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def paretoplot(x, k):
    """
    Create a Pareto quantile plot for regularly varying variables.

    Parameters:
    -----------
    x : array-like
        The input data.

    k : int
        The number of largest values to consider for the Pareto plot.

    Returns:
    --------
    reg : LinearRegression
        The fitted linear regression model.
    """
    # Select the k largest values from the input data
    xEx = x[x >= np.sort(x)[::-1][k-1]]
    k = len(xEx)

    # Compute the logarithm of the sorted values
    Lxord = np.log(np.sort(xEx))

    # Compute the indices for the Pareto plot
    inds = 1 - (np.arange(1, k + 1)) / (k + 1)
    negLinds = -np.log(inds)

    # Create the Pareto plot
    plt.figure(figsize=(10, 6))
    plt.scatter(negLinds, Lxord, label='Data points')
    plt.title(f"Pareto Quantile Plot, {k} Largest Values")

    # Fit a linear regression model to the data
    reg = LinearRegression().fit(negLinds.reshape(-1, 1), Lxord)
    coef = [reg.intercept_, reg.coef_[0]]

    # Plot the fitted line
    plt.plot(negLinds, reg.predict(negLinds.reshape(-1, 1)), 'r-', linewidth=2, label='Fitted line')

    # Add labels and legend
    plt.xlabel("Negative Log Indices")
    plt.ylabel("Log of Sorted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

    return reg

# Example usage:
# x = np.random.pareto(1, 1000) + 1  # Generate Pareto-distributed data
# paretoplot(x, 100)
