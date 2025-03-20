from matplotlib import pyplot as plt
import numpy as np 
import torch 

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

def compute_precision_recall_acc(val):
    tp, tn, fp, fn = val
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    return (precision *100., recall*100., accuracy*100.)

def generate_precision_recall_chart(vals): 
    assert len(vals) == 40, "vals should be of length 40"
    assert vals.shape[1] == 4, "vals should be of shape (40, 4)"

    vals_array = np.array(vals)  # shape will be (40, 4)

    # Number of attributes and categories
    num_attributes = vals_array.shape[0]
    categories = ['TP', 'TN', 'FP', 'FN']
    #categories = ['precision' , 'recall', "accuracy"]
    
    prec_rec_acc = np.array([compute_precision_recall_acc(val) for val in vals_array])
    precision = prec_rec_acc[:, 0]
    recall = prec_rec_acc[:, 1]
    accuracy = prec_rec_acc[:, 2]


    num_categories = len(categories)

    # x locations for each attribute group on the x-axis
    x = np.arange(num_attributes)

    # Width of each bar within a group (adjustable)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create a bar for each category in each group.
    ax.bar(x - 1.5*width, precision, width, label='precision')
    ax.bar(x - 0.5*width, recall, width, label='recall')
    ax.bar(x + 0.5*width, accuracy, width, label='accuracy')

    #ax.bar(x - 1.5*width, vals_array[:, 0], width, label='precision')
    #ax.bar(x - 0.5*width, vals_array[:, 1], width, label='recall')
    #ax.bar(x + 0.5*width, vals_array[:, 2], width, label='accuracy')
    #ax.bar(x + 0.5*width, vals_array[:, 2], width, label='FP')
    #ax.bar(x + 1.5*width, vals_array[:, 3], width, label='FN')

    # Set the x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels([f'Attr {i}' for i in range(num_attributes)], rotation=45)

    ax.set_ylabel('Performance')
    ax.set_title('Precision and Recall for Each Attribute')
    ax.legend()

    plt.tight_layout()
    plt.show()


def show_image(image):
    image = image.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    image = (image + 1) / 2  # Rescale to [0, 1]
    plt.imshow(image)
    plt.axis('off')
    plt.show()