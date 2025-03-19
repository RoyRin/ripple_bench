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
        print(f"features shape: {features.shape}")
    return features


def get_flattened_embedding(model: nn.Module, x: torch.Tensor, layer_ind: int):
    embeddings = []
    model_device = next(model.parameters()).device
    x = x.to(model_device)

    def hook(module, input, output):
        embeddings.append(output)

    layers = list(model.children())

    if layer_ind < 0 or layer_ind >= len(layers):
        raise ValueError(f"layer_num must be between 0 and {len(layers)-1}")

    handle = layers[layer_ind].register_forward_hook(hook)

    # Forward pass
    model(x)

    # Remove hook
    handle.remove()

    # Flatten the embedding
    #print(f"embeddings.shape - {embeddings[0].shape}")

    embedding = embeddings[0].detach().flatten(start_dim=1)

    return embedding

def get_embedding_size(model, layer_ind, input_dummy_shape = (1, 3, 224, 224)):
    """
    Get the size of the embedding from a specific layer of the model.
    Args:
        model: PyTorch model
        layer_ind: Index of the layer to extract the embedding from
    Returns:
        int: Size of the embedding
    """
    model_device = next(model.parameters()).device

    # Create a dummy input tensor (e.g., 1x3x224x224 for an image)
    dummy_input = torch.randn(input_dummy_shape).to(model_device)
    embedding_dummy = get_flattened_embedding(model, dummy_input, layer_ind=layer_ind)
    return embedding_dummy.shape[1]



def get_layer_count(model):
    """
    Get the number of layers in a model.
    Args:
        model: PyTorch model
    Returns:
        int: Number of layers in the model
    """
    return len(list(model.children()))


def set_up_probe_dataset(model, layer_ind, pos_indices, neg_indices, dataset, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = True):
    
    # Get embeddings for images without attribute
    pos_embeddings = []
    neg_embeddings = []
    print_every = len(pos_indices) // 4
    for ii, pos_ind in enumerate(pos_indices):
        if verbose and ii % print_every == 0:
            print("Processing sample", ii)

        pos_pt = dataset[pos_ind][0]
        img_tensor = pos_pt.unsqueeze(0).to(device)  # Add batch dimension
        #embedding = get_nth_to_last_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = get_flattened_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = embedding.squeeze(0)  # Remove batch dimension
        pos_embeddings.append(embedding)

    for ii, neg_pt in enumerate(neg_indices):
        if verbose and ii % print_every == 0:
            print("Processing sample", ii)
        neg_pt = dataset[neg_pt][0]
        img_tensor = neg_pt.unsqueeze(0).to(device)
        #embedding = get_nth_to_last_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = get_flattened_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
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


def __set_up_probe_dataset(model, layer_ind, pos_points, neg_points, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = True):
    
    # Get embeddings for images without attribute
    pos_embeddings = []
    neg_embeddings = []
    print_every = len(pos_points) // 4
    for ii, pos_pt in enumerate(pos_points):
        if verbose and ii % print_every == 0:
            print("Processing sample", ii)
        img_tensor = pos_pt.unsqueeze(0).to(device)  # Add batch dimension
        #embedding = get_nth_to_last_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = get_flattened_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = embedding.squeeze(0)  # Remove batch dimension
        pos_embeddings.append(embedding)

    for ii, neg_pt in enumerate(neg_points):
        if verbose and ii % print_every == 0:
            print("Processing sample", ii)
        img_tensor = neg_pt.unsqueeze(0).to(device)
        #embedding = get_nth_to_last_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = get_flattened_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
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
    

# train a 2 layer network on the probe dataset
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # single output for binary classification
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
    

def train_model(model_factory, dataset, labels, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs = 1000, verbose = True):
    dataset = torch.as_tensor(dataset)
    labels = torch.as_tensor(labels)

    dim = dataset.shape[1]
    X = dataset.float()
    labels = labels.float()
    
    
    labels = labels.to(device)
    X = X.to(device)
    # Define the loss function and the optimizer

    
    model = model_factory(input_dim= dim)
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
        if verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the trained model
    with torch.no_grad():
        predictions = (model(X).squeeze() >= 0.5).float()  # threshold at 0.5 for binary classification
        accuracy = (predictions == labels).float().mean()
        if verbose:
            print(f'Accuracy: {accuracy.item() * 100:.2f}%')

    return accuracy.item() * 100, model 

def train_logistic_regression(dataset, labels, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs = 1000, verbose= True):
    model_factory = lambda input_dim: LogisticRegression(input_dim= input_dim)
    return train_model(model_factory, dataset, labels, device = device, num_epochs = num_epochs, verbose=verbose)


def train_MLP(dataset, labels, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs = 1000, verbose= True):
    model_factory = lambda input_dim: MLP(input_dim= input_dim)
    return train_model(model_factory, dataset, labels, device = device, num_epochs = num_epochs, verbose=verbose)


def test_linear_probe(probe_dataset, probe_labels, model, num_epochs= 1000, verbose= True):
    
    model_device = next(model.parameters()).device

    probe_acc, probe_model = train_logistic_regression(probe_dataset, probe_labels, device = model_device, num_epochs = num_epochs, verbose=verbose)
    probe_tuple = (probe_acc, probe_dataset, probe_labels, probe_model)

    return probe_tuple

def test_MLP_probe(probe_dataset, probe_labels, model, num_epochs= 1000, verbose= True):
    model_device = next(model.parameters()).device

    probe_acc, probe_model = train_MLP(probe_dataset, probe_labels, device = model_device, num_epochs = num_epochs, verbose=verbose)
    probe_tuple = (probe_acc, probe_dataset, probe_labels, probe_model)

    return probe_tuple

def linear_probe(model, layer_ind, pos_indices, neg_indices, dataset, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs = 1000, verbose = True):
    """
    train a linear classifier on the features extracted from the specified layer,
    Returns:
         tuple 
    """ 
    # get the features from the specified layer
    probe_dataset, probe_labels = set_up_probe_dataset(model, layer_ind, pos_indices=pos_indices, neg_indices=neg_indices, dataset = dataset, device = device, verbose = verbose)
    if verbose:
        print(f"probe_dataset shape: {probe_dataset.shape}")

    return test_linear_probe(probe_dataset, probe_labels, model, num_epochs= num_epochs, verbose= verbose)

        
