import warnings
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN, MNIST


# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budget.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

SVHN_MEAN =(0.4376821, 0.4437697, 0.47280442)
SVHN_STD_DEV =(0.19803012, 0.20101562, 0.19703614)

# MNIST_MEAN =
# MNIST_STD_DEV = 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

transform_SVHN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD_DEV),
])


warnings.simplefilter("ignore")



def initialize_data_CIFAR10(batch_size=1024, DATA_ROOT= "../cifar10"):
    train_dataset = CIFAR10(root=DATA_ROOT,
                            train=True,
                            download=True,
                            transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
    )

    test_dataset = CIFAR10(root=DATA_ROOT,
                           train=False,
                           download=True,
                           transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def initialize_data_SVHN(batch_size=1024, DATA_ROOT= "../svhn"):
    train_dataset = SVHN(root=DATA_ROOT,
                         split='train',
                         download=True,
                         transform=transform_SVHN)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
    )

    test_dataset = SVHN(root=DATA_ROOT,
                        split='test',
                        download=True,
                        transform=transform_SVHN)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader



def initialize_data_MNIST(batch_size=1024, DATA_ROOT= "../mnist"):
    mnist_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5, ), (0.5, ),inplace=True),
    ])

    train_dataset = MNIST(root=DATA_ROOT,
                         train=True,
                         download=True,
                         transform=mnist_transform
                         )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
    )

    test_dataset = MNIST(root=DATA_ROOT,
                        train=False,
                        download=True,
                        transform=mnist_transform
                        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader