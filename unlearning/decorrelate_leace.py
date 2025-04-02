import torch
import numpy as np
from nltk.corpus import wordnet
import matplotlib.pyplot as plt

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def plot_sample_and_reconstruction(model, dataset, sample_idx=0):
    model.eval()  # Set the model to evaluation mode

    # Get a sample from the dataset
    original, _ = dataset[sample_idx]  # Ignore the label (if any)
    original = original.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Get the reconstruction
    with torch.no_grad():
        _, reconstruction = model(original)  # Pass through the autoencoder

    # Move data to CPU
    original_image = original.squeeze().cpu().numpy()  # Original shape: [1, 64, 64]
    reconstructed_image = reconstruction.squeeze().cpu().numpy()  # Flattened shape

    # Reshape if necessary
    if reconstructed_image.shape == (4096,):  # Check if flattened
        reconstructed_image = reconstructed_image.reshape(64, 64)  # Reshape to [64, 64]

    # Plot the images side by side
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.imshow(reconstructed_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("Reconstruction.pdf")
    plt.show()

    plt.tight_layout()
    plt.savefig("Reconstruction.pdf")
    plt.show()

def generate_data(n_samples_per_class = 400, n_features = 5, separation = 1): 
    """
    Creates a synthetic data set of size (2*n_samples_per_class, n_features) using random_seed
    """
        
    # negative examples
    X0 = np.random.randn(n_samples_per_class, n_features)
    y0 = -np.ones((n_samples_per_class,1))
    
    # postive examples
    shift = np.random.randn(1, n_features)
    X1 = np.random.randn(n_samples_per_class, n_features) + separation * shift
    y1 = np.ones((n_samples_per_class,1))
    
    X_train = np.vstack((X0,X1))
    y_train = np.vstack((y0,y1)).reshape(-1)

    # negative examples
    X0 = np.random.randn(n_samples_per_class, n_features)
    y0 = -np.ones((n_samples_per_class,1))
    
    # postive examples
    X1 = np.random.randn(n_samples_per_class, n_features) + separation * shift
    y1 = np.ones((n_samples_per_class,1))

    X_test = np.vstack((X0,X1))
    y_test = np.vstack((y0,y1)).reshape(-1)
    
    return X_train, X_test, y_train, y_test

def plot_classes(data, labels, ax=None, x_label = None, y_label = None):
    """
    Scatter plot for two-dimensional data with labels in {-1, 1}.
    
    Parameters:
    - data: A 2D NumPy array of shape (n_samples, 2), where each row is a point (x1, x2).
    - labels: A 1D NumPy array of shape (n_samples,) with labels in {-1, 1}.
    """
    # Separate data by labels
    class1_data = data[labels == 1]
    class_neg1_data = data[labels == -1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the scatter plot
    ax.scatter(class1_data[:, 0], class1_data[:, 1], marker='o', color='blue', label='Class 1 (Label = 1)')
    ax.scatter(class_neg1_data[:, 0], class_neg1_data[:, 1], marker='x', color='red', label='Class 2 (Label = -1)')
    
    # Add labels and legend
    ax.set_xlabel("Feature 1") if x_label == None else ax.set_xlabel(x_label)
    ax.set_ylabel("Feature 2") if y_label == None else ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True)
    plt.show()

def decorrelate(X, Y):
    """
    Erase the component of X that can be predicted from Y by computing
    the sample means and covariances.
    
    Parameters:
    -----------
    X : np.ndarray or torch.Tensor
        Matrix of X instances, shape (n_samples, d_x).
    Y : np.ndarray or torch.Tensor
        Matrix (or vector) of Y instances, shape (n_samples, d_y) or (n_samples,).
    
    Returns:
    --------
    X_erased : np.ndarray
        The residuals X - E[X|Y], where E[X|Y] is computed from the sample
        means and covariances.
    """
    # Convert torch tensors to numpy arrays if necessary.
    if hasattr(X, 'cpu'):
        X = X.cpu().detach().numpy()
    if hasattr(Y, 'cpu'):
        Y = Y.cpu().detach().numpy()
    
    # If Y is one-dimensional, reshape it to be a column vector.
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n_samples = X.shape[0]
    
    # Compute sample means
    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)
    
    # Center the data
    X_centered = X - mean_X
    Y_centered = Y - mean_Y
    
    # Compute sample covariances (using unbiased estimator: divide by n-1)
    cov_XY = (X_centered.T @ Y_centered) / (n_samples - 1)
    cov_Y = (Y_centered.T @ Y_centered) / (n_samples - 1)
    
    # Compute the linear mapping A = cov_XY @ inv(cov_Y)
    A = cov_XY @ np.linalg.pinv(cov_Y)
    
    # Compute the predicted value E[X|Y] = mean_X + A * (Y - mean_Y)
    X_pred = mean_X + (A @ Y_centered.T).T
    
    # Subtract the predictable component from X to obtain the residuals.
    X_erased = X - X_pred
    return X_erased

def get_synonyms(word):
    """
    Return a set of synonyms for the given word using WordNet.
    """
    synonyms = {word.lower()}
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return synonyms

def erase_by_ablation(weights, concept_to_erase, vocabulary):
    """
    Erases the component corresponding to a specific concept (or its synonyms)
    from SpLiCE weights.
    
    Parameters:
    -----------
    weights : torch.Tensor or np.ndarray
        The sparse weight vector output by SpLiCE (shape: [vocab_size]).
    concept_to_erase : str
        The concept (token) that should be ablated (erased) from the weight vector.
    vocabulary : list of str
        The list of tokens corresponding to each index in the weight vector.
    
    Returns:
    --------
    weights_erased : same type as `weights`
        A copy of the input weights with entries corresponding to the concept (or its synonyms) set to 0.
    """
    # Expand concept_to_erase into a set of synonyms
    synonyms = get_synonyms(concept_to_erase)
    print(synonyms)
    
    if torch.is_tensor(weights):
        weights_erased = weights.clone()
    else:
        weights_erased = np.copy(weights)
    
    found = False
    # Loop over the vocabulary and zero out weights for tokens that match any synonym.
    for idx, token in enumerate(vocabulary):
        if token.lower() in synonyms:
            weights_erased[idx] = 0.0
            found = True
    
    if not found:
        print(f"Warning: None of the synonyms for '{concept_to_erase}' were found in the vocabulary.")
    
    return weights_erased

class LEACE:
    def __init__(self):
        self.whitening_mat_ = None  # Whitening matrix
        self.projection_mat_ = None # Projection matrix
        self.mean_X_ = None  # Mean of the X data
        self.mean_Y_ = None  # Mean of the Y data
        self.fitted = False

    def fit(self, X, Y, unbiased_estimator = True):
        """
        Fit the LEACE model by approximating

        Parameters:
        - X : np.ndarray of shape (n_samples, n_features_x)
          The data from which you want to erase the concept.
        - Y : np.ndarray of shape (n_samples, n_features_y)
          The concept values.

        Returns:
        - self : Returns an instance of self.
        """
        n_samples = X.shape[0]
        dim_x = X.shape[1]
        # Compute means
        self.mean_X_ = np.mean(X, axis=0)
        if Y.ndim == 1: # Ensure Y is stored as a matrix
            Y = Y.reshape(-1, 1) 
        self.mean_Y_ = np.mean(Y, axis=0)
        
        # Center the data
        X_centered = X - self.mean_X_
        Y_centered = Y - self.mean_Y_
        
        # Compute covariance matrices (using unbiased estimator: divide by n-1)
        if unbiased_estimator:
            cov_XY = (X_centered.T @ Y_centered) / (n_samples - 1)
            cov_XX = (X_centered.T @ X_centered) / (n_samples - 1)
        else:
            cov_XY = (X_centered.T @ Y_centered) / n_samples
            cov_XX = (X_centered.T @ X_centered) / n_samples

        # Compute whitening matrix using eigendecomposition
        eigenvalues, Q = np.linalg.eigh(cov_XX) # using eigh because cov_xx is symmetric
        W_plus = Q @ np.diag(np.sqrt(eigenvalues)) @ Q.T # inverse of whitening matrix
        W = np.linalg.pinv(W_plus) # whitening matrix 
        P_W_XZ = (W @ cov_XY) @ np.linalg.pinv(W @ cov_XY)

        # Compute oblique projection matrix
        self.projection_mat_ = np.eye(dim_x) - W_plus @ P_W_XZ @ W
        self.fitted = True
        return self

    def transform(self, X):
        """
        Transform the data X to linearly guard Y using the oblique projection matrix P.

        Parameters:
        - X : np.ndarray of shape (n_samples, n_features)
          The data to be transformed.

        Returns:
        - X_erased : np.ndarray of shape (n_samples, n_features)
          The transformed data with the concept component removed.
        """
        if not self.fitted:
            raise ValueError("The LEACE model is not fitted yet. Please call 'fit' first.")
        
        # Center data
        X_centered = X - self.mean_X_
        
        return X @ self.projection_mat_.T 

    def fit_transform(self, X, Y):
        """
        Convenience method that fits the model and transforms X in one call.

        Parameters:
        - X : np.ndarray of shape (n_samples, n_features)
        - Y : np.ndarray of shape (n_samples,) or (n_samples, 1)

        Returns:
        - X_erased : np.ndarray of shape (n_samples, n_features)
        """
        return self.fit(X, Y).transform(X)