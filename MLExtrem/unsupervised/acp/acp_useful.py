#Useful ACP
## PA naming modifié, plus explicite et uniforme avec ppareto et q pareto, fct validée
import numpy as np

def hillestim(sample_data, num_extremes):  # Hill estimator
    sorted_data = np.sort(sample_data)[::-1]  # Sort in decreasing order
    hill_estimate = 1 / num_extremes * np.sum(np.log(sorted_data[:num_extremes] / sorted_data[num_extremes]))
    standard_deviation = 1 / np.sqrt(num_extremes) * hill_estimate
    return {"test": hill_estimate, "sdev": standard_deviation}


def rpareto(num_samples, alpha):  # Generate samples from Pareto distribution
    uniform_samples = np.random.uniform(size=num_samples)
    pareto_samples = (1 - uniform_samples)**(-1 / alpha)
    return pareto_samples


def qpareto(prob, alpha):
    return (1 - prob)**(-1/alpha)

def ppareto(x, alpha, scale):
    return (x/scale)**(-alpha)
