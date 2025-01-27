import torch
import torchvision
import torchvision.transforms as transforms
# get mnist dataset
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

import matplotlib.pyplot as plt

def visualize_mnist_image(image_matrix):
    """
    Visualizes a 28x28 matrix as an image.
    
    Parameters:
    - image_matrix: A 28x28 numpy array representing the image.
    """
    if len(image_matrix.shape) == 1:
        sqrt = int(np.sqrt(image_matrix.shape[0]))

        image_matrix = image_matrix.reshape(sqrt, sqrt)

    plt.imshow(image_matrix, cmap='gray')  # Use grayscale color map for visualization
    plt.colorbar()  # Optional: Adds a colorbar to show the mapping of intensity values
    plt.show()


def crop(x, cropper_x = 4, cropper_y = 4):

    ret = x.reshape(28,28)[cropper_x:28-cropper_x, cropper_y:28-cropper_y].flatten()
    return ret

def add_noise(x, range_=5):
    # add random noise to the image, 
    v = x + np.random.random_integers(-range_, range_, size = x.shape)
    # clip the values to be between 0 and 255
    v = np.clip(v, 0, 255)
    return v

def get_mnist_2_dataset():
    # Load the MNIST dataset

    mnist = fetch_openml('mnist_784', version=1)

    # Extract images and labels
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)  # Convert labels to integers

    # Filter out images for digits 0 and 1 only
    is_class_0_or_1 = (y == 0) | (y == 1)
    X_filtered = X[is_class_0_or_1]
    y_filtered = y[is_class_0_or_1]

    # Split the filtered dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)
    # Scale the features
    if False:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.astype(np.float64))
        X_test = scaler.transform(X_test.astype(np.float64))
    X_train= X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    return X_train, y_train, X_test, y_test

def get_mnist_2_dataset_well_conditioned(noise_addition = 30):
    X_train, y_train, X_test, y_test = get_mnist_2_dataset()
    # crop and add noise
    
    X_train = np.array([add_noise(crop(x), range_ = noise_addition) for x in X_train])

    X_test = np.array([add_noise(crop(x), range_ = noise_addition) for x in X_test])
    return X_train, y_train, X_test, y_test



if False:

    def get_dataloader(dataset, batch_size=256, num_workers=8, shuffle=False):

        return torch.utils.data.DataLoader(
            dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
        )


    def get_mnist_datasets():
        """
        Returns the MNIST dataset.
        Returns:
        - Tuple: The MNIST dataset.
        """
        # Load the MNIST dataset
        mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        return mnist, mnist_test


    def get_mnist_dataloader(
        batch_size=256,
        num_workers=8,
        split="train",
        shuffle=False,
        augment=True,
        indices=None,
    ):
        dataset, test_ds = get_mnist_datasets(split=split, augment=augment, indices=indices)

        loader = torch.utils.data.DataLoader(
            dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_ds, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
        )
        return loader, test_loader

    ####
    ####
    ####

    def get_mnist_2_datasets(classes = [0,1]):
        """
        get mnist dataset but only for 2 classes
        """
        mnist, mnist_test = get_mnist_datasets()
        indices = []
        for i in range(len(mnist)):
            if mnist[i][1] in classes:
                indices.append(i)
        mnist = torch.utils.data.Subset(mnist, indices)
        indices = []
        for i in range(len(mnist_test)):
            if mnist_test[i][1] in classes:
                indices.append(i)
        mnist_test = torch.utils.data.Subset(mnist_test, indices)
        return mnist, mnist_test



    def get_mnist_2_dataloaders(classes = [0,1], batch_size=256, num_workers=8, shuffle=False):
        """
        get mnist dataloader but only for 2 classes
        """
        mnist, mnist_test = get_mnist_2_datasets(classes)
        loader = get_dataloader(mnist, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        test_loader = get_dataloader(mnist_test, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        
        return loader, test_loader