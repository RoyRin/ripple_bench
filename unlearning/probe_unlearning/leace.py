import torch 


def whiten_data_torch(X, epsilon=1e-5):
    """
    Whitens the dataset X using eigenvalue decomposition in PyTorch.
    
    The input X should be a tensor of shape (n_samples, n_features). The function centers
    the data and applies the whitening transform so that the output has an (approximately)
    identity covariance matrix.
    
    Parameters:
    - X : torch.Tensor
          Input data of shape (n_samples, n_features).
    - epsilon : float, optional
          Small constant added for numerical stability.
    
    Returns:
    - X_whitened : torch.Tensor
          The whitened data with approximately unit covariance.
    """
    # Center the data (zero mean)
    X_centered = X - torch.mean(X, dim=0, keepdim=True)
    
    # Compute the covariance matrix: (n_features x n_features)
    n_samples = X_centered.size(0)
    cov = torch.mm(X_centered.T, X_centered) / (n_samples - 1)
    
    # Eigen-decomposition of the covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Create the diagonal matrix for the inverse square root of eigenvalues
    D_inv = torch.diag(1.0 / torch.sqrt(eigenvalues + epsilon))
    
    # Compute the whitening matrix
    whitening_matrix = torch.mm(eigenvectors, torch.mm(D_inv, eigenvectors.T))
    
    # Apply the whitening transformation
    X_whitened = torch.mm(X_centered, whitening_matrix)
    
    return X_whitened
