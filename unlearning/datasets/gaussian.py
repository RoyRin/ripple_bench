import torch 


def generate_powerlaw_gaussian_dataset(n_samples, n_features, mean = None, alpha=2.0, epsilon=1e-5, device=torch.device("cpu"), seed=42):
    """
    Generates a Gaussian dataset with a power law diminishing covariance matrix.
    
    """

    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    
    # Create a vector of variances that decay with power law.
    # Variance for feature i: 1/(i+1)^alpha + epsilon
    indices = torch.arange(1, n_features + 1, dtype=torch.float32, device=device)
    variances = 1.0 / (indices ** alpha) + epsilon
    
    # Build the diagonal covariance matrix
    cov_matrix = torch.diag(variances)
    
    # Define the zero mean for all features
    if mean is None:
        mean = torch.zeros(n_features, device=device)
    elif type(mean) == float:
        # mean is a float, create a mean vector of the same value
        mean = torch.ones(n_features, device=device) * mean


    
    # Create the multivariate normal distribution and sample data
    distribution = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_matrix)
    data = distribution.sample((n_samples,))
    
    return data, cov_matrix