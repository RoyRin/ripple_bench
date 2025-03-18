from matplotlib import pyplot as plt
import torch


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
# Create a simple synthetic dataset
# We'll create two clusters in 2D space:
# - Cluster for class 0 centered at (-1, -1)
# - Cluster for class 1 centered at (1, 1)
torch.manual_seed(42)




def get_nth_to_last_embedding(model, x, layer_ind=-1):
    """
    Extract embedding from the n-th to last layer of ResNet.
    Args:
        model: ResNet model (e.g., ResNet50)
        x: Input image tensor (batch_size, 3, H, W)
        n: Number of layers to remove from the end (default: 2 for second-to-last layer)
    Returns:
        features: Extracted feature tensor
    """
    # Get model layers except last n layers
    truncated_model = nn.Sequential(*list(model.children())[:layer_ind])
    
    with torch.no_grad():
        features = truncated_model(x)  # Pass input through the truncated model
        features = features.squeeze(-1).squeeze(-1)  # Remove unnecessary dimensions

    return features

def get_layer_count(model):
    """
    Get the number of layers in a model.
    Args:
        model: PyTorch model
    Returns:
        int: Number of layers in the model
    """
    return len(list(model.children()))


def set_up_probe_dataset(model, layer_ind, pos_points, neg_points, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    # Get embeddings for images without attribute
    pos_embeddings = []
    neg_embeddings = []
    for ii, pos_pt in enumerate(pos_points):
        if ii % 50 == 0:
            print("Processing sample", ii)
        img_tensor = pos_pt.unsqueeze(0).to(device)  # Add batch dimension
        embedding = get_nth_to_last_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = embedding.squeeze(0)  # Remove batch dimension
        pos_embeddings.append(embedding)

    for ii, neg_pt in enumerate(neg_points):
        if ii % 50 == 0:
            print("Processing sample", ii)
        img_tensor = neg_pt.unsqueeze(0).to(device)
        embedding = get_nth_to_last_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = embedding.squeeze(0)  # Remove batch dimension
        neg_embeddings.append(embedding)


    all_embeddings = pos_embeddings + neg_embeddings
    # stack
    #all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_embeddings = torch.stack(all_embeddings)
    #all_embeddings = torch.from_numpy(all_embeddings)
    labels_ = torch.cat([torch.ones(len(pos_embeddings)), torch.zeros(len(neg_embeddings))])

    #labels_ = np.concatenate([np.ones(len(pos_embeddings)), np.zeros(len(neg_embeddings))])
    return all_embeddings, labels_



# Define the Logistic Regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # single output for binary classification

    def forward(self, x):
        # Apply linear transformation followed by a sigmoid activation
        x= x.float()
        return torch.sigmoid(self.linear(x))
    
    
def train_logistic_regression(dataset, labels, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs = 1000):
    dataset = torch.as_tensor(dataset)
    labels = torch.as_tensor(labels)

    dim = dataset.shape[1]


    print(f"probe_dataset shape: {dataset.shape}")
    X = dataset.float()
    labels = labels.float()
    
    
    labels = labels.to(device)
    X = X.to(device)
    # Define the loss function and the optimizer



    model = LogisticRegression(dim)
    # model to device
    model.to(device)

    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass: compute predicted y by passing X to the model
        outputs = model(X).squeeze()  

        loss = criterion(outputs, labels)
        
        # Zero gradients, perform backward pass, and update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the trained model
    with torch.no_grad():
        predictions = (model(X).squeeze() >= 0.5).float()  # threshold at 0.5 for binary classification
        accuracy = (predictions == labels).float().mean()
        print(f'Accuracy: {accuracy.item() * 100:.2f}%')

    return accuracy.item() * 100, model 

def linear_probe(model, layer_ind, pos_points, neg_points, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs = 1000):
    """
    train a linear classifier on the features extracted from the specified layer,
    Returns:
         the accuracy of the classifier on the validation set
    """ 
    
    # get the features from the specified layer
    probe_dataset, probe_labels = set_up_probe_dataset(model, layer_ind, pos_points, neg_points, device = device)
    print(f"probe_dataset shape: {probe_dataset.shape}")
    return train_logistic_regression(probe_dataset, probe_labels, device = device, num_epochs = num_epochs)



        
