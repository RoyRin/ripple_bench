from matplotlib import pyplot as plt
import numpy as np 


def plot_top2_pcs_torch(data, title="Projection onto Top 2 Principal Components", labels=None):
    """
    Computes the top two principal components using PyTorch and plots the data
    projected onto these components.
    """
    # Center the data (zero mean)
    data_centered = data - data.mean(dim=0, keepdim=True)
    
    # Compute the covariance matrix (n_features x n_features)
    n_samples = data_centered.shape[0]
    cov_matrix = torch.mm(data_centered.T, data_centered) / (n_samples - 1)
    
    # Eigen-decomposition of the covariance matrix (eigenvalues in ascending order)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # Select the top 2 eigenvectors (columns corresponding to the largest eigenvalues)
    top2_eigenvectors = eigenvectors[:, -2:]  # shape: (n_features, 2)
    
    # Project the centered data onto the top 2 principal components
    projected_data = torch.mm(data_centered, top2_eigenvectors)  # shape: (n_samples, 2)
    
    # Convert to numpy arrays for plotting (ensure data is on CPU)
    projected_np = projected_data.cpu().numpy()
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels
    else:
        labels_np = None
    
    # Plot the data
    plt.figure(figsize=(8, 6))
    if labels_np is None:
        plt.scatter(projected_np[:, 0], projected_np[:, 1], alpha=0.5)
    else:
        scatter = plt.scatter(projected_np[:, 0], projected_np[:, 1],
                              c=labels_np, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label="Labels")
    
    # make xlim and ylim equal
    x_max = projected_np[:, 0].max()
    x_min = projected_np[:, 0].min()
    y_max = projected_np[:, 1].max()
    y_min = projected_np[:, 1].min()
    max_ = max(abs(x_max), abs(x_min), abs(y_max), abs(y_min))
    min_ = min(abs(x_max), abs(x_min), abs(y_max), abs(y_min))
    plt.xlim(-max_, max_)
    plt.ylim(-max_, max_)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.grid(True)
    plt.show()
