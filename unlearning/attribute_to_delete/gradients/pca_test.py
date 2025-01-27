import logging
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import local_models
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from datasets import (initialize_data_CIFAR10, initialize_data_MNIST,
                      initialize_data_SVHN)
from scipy.sparse.linalg import svds as scipy_sparse_svds
from sklearn.utils.extmath import randomized_svd
# from opacus.utils.batch_memory_manager import BatchMemoryManager
from torchvision import models
from tqdm import tqdm

###
#
# Todo: 1. see if you can run it locall on mac
# 2. see how many gradients are computed in the gradient dictionary
#   okay, confirmed -> number of gradients computed = number of dimensions
# 3. see if it's possible to create a continguous memory numpy (it must be)
# 4. get the sizes,
# 5. run SVD
#
#


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

warnings.simplefilter("ignore")


def write_yaml(d, path):
    with open(path, 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)

def get_model_size(model):
    return sum(p.numel() for p in model.parameters())

def accuracy(preds, labels):
    return (preds == labels).mean()


# dictionary
def compute_svd_of_aggregated_layer_gradients(gradients, k):
    """
    takes in a dictionary of gradients and computes the SVD of each layer
    :param gradients: dictionary of gradients
    :param k: number of singular values to compute
    :return: dictionary of layer names and their corresponding singular values
    """

    aggregated_gradients = {}
    for name, grads in gradients.items():
        # Check if gradients are more than 2 dimensions and flatten
        if grads[0].ndim > 2:
            flattened_grads = [g.view(g.size(0), -1) for g in grads]
        elif grads[0].ndim == 1:
            # If the gradient is 1D (e.g., bias), add an extra dimension to make it 2D
            flattened_grads = [g.unsqueeze(1) for g in grads]
        else:
            flattened_grads = grads
        # Stack along a new dimension (0) since all gradients are now 2D
        aggregated_gradients[name] = torch.stack(flattened_grads, dim=0)
    
    print(f"flattened sizes")
    total_len = 0
    for k_, v_ in aggregated_gradients.items():
        matrix = v_
        if v_.ndim > 2:
            matrix = matrix.reshape(-1, matrix.shape[0])  
            #print(f"--- {v_.shape} - > {matrix.shape}")
        matrix = matrix.cpu().numpy()
        print(f">>> {k_}: {matrix.shape}")
        total_len += matrix.shape[0]
    print(f"total length - {total_len}")
    arr = np.empty((total_len, matrix.shape[1])) # should be d x n dimensional, for n gradients
    n = matrix.shape[1]
    print(f"arr.sape - >{arr.shape}")
    
    #
    ## populate Gradient Matrix
    #
    i =0
    for k_, v_ in aggregated_gradients.items():
        matrix = v_
        if v_.ndim > 2:
            matrix = matrix.reshape(-1, matrix.shape[0])  
            #print(f"--- {v_.shape} - > {matrix.shape}")
        matrix = matrix.cpu().numpy()
        layer_d = matrix.shape[0]
        # NOTE: this is probably inccorectly slicing in `matrix` into `arr`.
        # TODO - pick it back up here.
        for pt_ind in range(n):
            arr[i:i+layer_d][pt_ind] = matrix[pt_ind]
        
        i+= layer_d
    print(f"populated entire SVD matrix  {arr.shape}")

    # TODO -> pipe thiese into a big matrix
    # TODO -> compute SVD on this.
    import sys
    sys.exit()

    svd_results = {}
    for name, agg_grads in aggregated_gradients.items():
        matrix = aggs_grads
        if agg_grads.ndim > 2:
            matrix = agg_grads.reshape(-1, agg_grads.shape[0])  

        matrix = matrix.cpu().numpy()
        print(f"name is {name} - matrix.shape - {matrix.shape}")
        # Compute SVD
        try:
            U, s, Vh = randomized_svd(matrix, n_components=k , random_state=0)
            svd_results[name] = s
            # print(size of s)
            print(f"s length - {len(s)}")
        except Exception as e:
            print(f"Error computing SVD for {name}: {e}")
    return svd_results

# dictionary
def get_layer_gradient_SVDs_wrapper(model, train_loader, train_loader_bs_1, optimizer, device, k=200, num_gradients = 10):
    """
    wrapper which first computes the aggregated gradients and then computes the SVD
    """
    model.train()
    torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()
    gradients = {name: [] for name, _ in model.named_parameters()}
    torch.cuda.empty_cache()
    for i, (images, target) in enumerate(train_loader_bs_1):
        optimizer.zero_grad()
        images = images.to(device)
        target = target.to(device)
        # compute output
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        # do not update the model, do not step
        #optimizer.step()
        for name, param in model.named_parameters():
            # BUG - note, observe that these param.grad are nmot flattened.
            # as such thery can be of the shape
            # `layer3.0.downsample.0.weight - torch.Size([256, 128, 1, 1])`
            if param.requires_grad and param.grad is not None:
                # Detach the gradient and store it
                gradients[name].append(param.grad.detach().clone())
        if i >= num_gradients:
            break  
    # compute SVD on the gradients in `gradients`
    print(f"completed gradient SVDs")


    return compute_svd_of_aggregated_layer_gradients(gradients, k=k)

##############
# Compute Gradients as an array - not a dictionary
#############
def get_gradient_SVDs_wrapper(model, train_loader, train_loader_bs_1, optimizer, device, k=200, num_gradients = 100):
    """
    wrapper which first computes the aggregated gradients and then computes the SVD
    """
    model.train()
    torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()

    #gradients = {name: [] for name, _ in model.named_parameters()}
    model_size = get_model_size(model)
    gradients = np.empty((num_gradients, model_size))

    torch.cuda.empty_cache()
    for i, (images, target) in enumerate(train_loader_bs_1):
        if i >= num_gradients:
            break        
        optimizer.zero_grad()
        images = images.to(device)
        target = target.to(device)
        # compute output
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        # do not update the model, do not step -- optimizer.step()
        gradient = torch.cat([param.grad.view(-1) for param in model.parameters()])
        gradients[i] = gradient.cpu().numpy()
        
    # compute SVD on the gradients in `gradients`
    return compute_svd_of_aggregated_gradients(gradients, k=k)

def compute_svd_of_aggregated_gradients(gradients, k):
    """
    takes in a dictionary of gradients and computes the SVD of each layer
    :param gradients: dictionary of gradients
    :param k: number of singular values to compute
    :return: dictionary of layer names and their corresponding singular values
    """
    svd_results = None
    try:
        _, s, _ = randomized_svd(gradients, n_components=k , random_state=0)
        svd_results = s
        # print(size of s)
        print(f"s length - {len(s)}")
    except Exception as e:
        print(f"Error computing SVD : {e}")
    return svd_results

#####

def train(
        model,
        train_loader,
        optimizer,
        epoch,
        device,
        train_loader_bs_1= None,
        k=20, desired_num_batches= 40):
    model.train()
    torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()
    losses = []
    top1_acc = []
    SVDs = {}
    num_batches = len(train_loader)
    # Initialize a dictionary to hold lists of gradients for each parameter
    gradients = {name: [] for name, _ in model.named_parameters()}

    torch.cuda.empty_cache()
    for i, (images, target) in enumerate(train_loader):
        print(f"i- {i}")
        optimizer.zero_grad()
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        # measure accuracy and record loss
        acc = accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc)

        loss.backward()

        optimizer.step()  # add the noise to the gradients
        #print(f"i is {i}")
        if (i + 1) % 200 == 0:
            print(f"Train Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f}")
        
        if False:
            if i >= num_batches - desired_num_batches:
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Detach the gradient and store it
                        gradients[name].append(param.grad.detach().clone())
        # HACK
        if i >= 10:
            break 
    # compute SVD on the gradients in `gradients`
    # SVDs = compute_svd_of_aggregated_gradients(gradients, k=k)
    print(f"trained one epoch")

    #SVDs = get_gradient_SVDs_wrapper(model, train_loader,train_loader_bs_1, optimizer, device, k=k)
    SVDs = get_layer_gradient_SVDs_wrapper(model, train_loader,train_loader_bs_1, optimizer, device, k=k)
    

    print(f"Train Epoch: {epoch} \t"
          f"Loss: {np.mean(losses):.6f} "
          f"Acc@1: {np.mean(top1_acc) * 100:.6f}")

    # HACK - NANMEAN
    return np.mean(top1_acc), np.mean(losses), SVDs


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(f"\tTest set:"
          f"Loss: {np.mean(losses):.6f} "
          f"Acc: {top1_avg * 100:.6f} ")
    return np.mean(top1_acc)



def initialize_model_CNN(device, LR=1e-2, momentum=0.9):
    model = local_models.CIFAR10_CNN()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum)
    return model, criterion, optimizer


def initialize_model_resnet18(device, LR=1e-2, momentum=0.9):
    model = models.resnet18(num_classes=10)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum)
    return model, criterion, optimizer


def initialize_model_WRN(device, LR=1e-2):
    # model = models.resnet18(num_classes=10)
    model = torch.hub.load(
        'pytorch/vision:v0.10.0',
        'wide_resnet50_2',
        pretrained=False  # , force_reload= True
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LR)

    return model, criterion, optimizer


if __name__ == "__main__":
    print(f"first line")
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    hub_dir = Path(torch.hub.get_dir()) / f"checkpoints_pca"
    torch.hub.set_dir(hub_dir)
    dataset_name = "cifar-10"
    # dataset_name = "mnist"
    # dataset_name = "svhn"i
    # model_name = "handcrafted_CNN"
    
    SVD_DIR = Path("/n/home04/rrinberg/code/unlearning-with-trak/unlearning/gradients/SVDs")
    SVD_DIR_mac = Path("/Users/roy/code/research/unlearning-with-trak/unlearning/gradients")
    if SVD_DIR_mac.exists():
        SVD_DIR = SVD_DIR_mac 
    # make dir
    SVD_DIR.mkdir(parents=True, exist_ok=True)

    model_name = "CNN"  # WRN, resnet18
    model_name = "resnet18"
    EPOCHS = 50  # 100
    trials = 1

    momentum = 0.9
    batch_size = 128
    LR = .1
    momentum = 0.9
    svd_k = 40 # how many singular values you compute
    desired_num_batches = 25 # how many batches you compute the SVD over
    
    start_time = datetime.now()


    print(f"we are starting the PCA experiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dataset_name == "svhn":
        data_loader_f = initialize_data_SVHN
    elif dataset_name == "mnist":
        data_loader_f =  initialize_data_MNIST
    elif dataset_name == "cifar-10":
        data_loader_f = initialize_data_CIFAR10
    else: 
        raise Exception(f"don't use : {dataset_name}")
    
    train_loader, test_loader = data_loader_f(
            batch_size=batch_size)
    
    train_loader_bs_1, test_loader_bs_1 = data_loader_f(
            batch_size=10)
        
    print(f"dataset - {dataset_name}")
    print(f"starting experiment")
    results = []
    save_path = f"{dataset_name}__experiment_{model_name}__batch_size_{batch_size}__LR_{LR}"
    
    for trial in range(trials):

        # initialize_model
        if model_name == "WRN":

            model, criterion, optimizer = initialize_model_WRN(
                device=device,
                LR=LR)
        elif model_name == "CNN":
            model, criterion, optimizer = initialize_model_CNN(
                device=device,
                # train_loader=train_loader,
                # bn_noise_multiplier=8,
                LR=LR,
                momentum=momentum)
        elif model_name == "resnet18":
            model, criterion, optimizer = initialize_model_resnet18(
                device=device,
                # train_loader=train_loader,
                # bn_noise_multiplier=8,
                LR=LR,
                momentum=momentum)

        else:
            raise Exception(f"don't use : {model_name}")

        print(f"model is {model_name}")
        model_size = get_model_size(model)

        print(f"model size - {model_size}")

        accs, test_accs, losses, SVDs = [],  [], [], []

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10,
            gamma=1.)  # 0.5) # every 10 steps, halve it

        # TODO: if accuracy drops for 10 epochs in arow, end
        

        for epoch in range(EPOCHS):
            
            print(f"training on epoch {epoch}")
            logging.info(f"training on epoch {epoch}")
            top1_acc, loss, svds_ = train(
                model,
                train_loader,
                optimizer,
                epoch + 1,
                device,
                k=svd_k,
                train_loader_bs_1=train_loader_bs_1,
                desired_num_batches= desired_num_batches)
            
            # cast SVDs to floats
            for k, v in svds_.items():
                svds_[k] = [v_i.tolist() for v_i in v]
            # change epoch str to be zero padded

            svd_pkl = SVD_DIR / f"SVD__{dataset_name}__{model_name}_{model_size}__{str(epoch).zfill(3)}__{trial}__{date_str}.pkl"
            #

            with open(svd_pkl, 'wb') as handle:
                pickle.dump(svds_, handle, protocol=pickle.HIGHEST_PROTOCOL)

            scheduler.step()
            print(f"completed training epoch {epoch} ")

            accs.append(float(top1_acc))
            losses.append(float(loss))
            print(f"{epoch} - train acc : {top1_acc}")
            logging.info(top1_acc)

            top1_acc__test = float(
                test(model, test_loader, device))
            test_accs.append(top1_acc__test)
            SVDs.append(svds_)
            single_epoch = {
                "train_acc": accs,
                "test_acc": test_accs,
                "train_loss": losses,
                 "LR": LR,
                "batch_size": batch_size,
                "model_name": model_name,
                "momentum": momentum,
                "dataset_name": dataset_name,
                "SVDs" : SVDs
            }
            results.append(single_epoch)

            write_yaml(single_epoch, SVD_DIR / f"{save_path}__{date_str}__temp.yaml")

        single_epoch = {
            "train_acc": accs,
            "test_acc": test_accs,
            "train_loss": losses,
            # "test_acc": top1_acc
            "LR": LR,
            "batch_size": batch_size,
            "model_name": model_name,
            "momentum": momentum,
            "dataset_name": dataset_name,
            "SVDs" : SVDs
        }
        write_yaml(results, SVD_DIR/ f"{save_path}__{date_str}.yaml")
        time_elapsed = datetime.now() - start_time
        print(f"Time elapsed: {time_elapsed}")


############################################################
############################################################
############################################################

def __get_gradients(optimizer, SVDs, k):
    for param_i, p in enumerate(optimizer.params):
        summed_grad = p.summed_grad.to("cpu")
        if param_i not in SVDs:
            SVDs[param_i] = []
        if (len(p.shape) > 1):
            # if its more than 1000 dimensional
            print(f"summed_grad.shape - {summed_grad.shape}")
            # if (p.shape[0] < 1000):
            #    S = torch.linalg.svdvals(summed_grad)
            # else:
            if True:
                u, s, vt = scipy_sparse_svds(
                    summed_grad, k=k, return_singular_vectors=False)
                S = s
            SVDs[param_i].append(S)


def get_gradients(optimizer, SVDs, k):
    # Example modification to your loop
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            # Assuming you want to sum gradients along dim=0
            summed_grad = p.grad.data.sum(dim=0).to("cpu")
            param_i = id(p)  # Use the parameter's ID as a unique identifier
            if param_i not in SVDs:
                SVDs[param_i] = []
            if len(p.shape) > 1:
                print(f"summed_grad.shape - {summed_grad.shape}")
                # Check the scat ize of the parameter
                # if p.shape[0] < 1000:
                #    S = torch.linalg.svdvals(summed_grad)
                # else:
                if True:
                    # Assuming you have already imported scipy.sparse.linalg.svds as scipy_sparse_svds
                    # and defined k
                    u, s, vt = scipy_sparse_svds(
                        summed_grad, k=k, return_singular_vectors="u")
                    S = s
                SVDs[param_i].append(S)



def __compute_svd_of_gradients(model, k):
    """
    Compute the SVD of gradients for all trainable parameters of the model.

    :param model: PyTorch model with computed gradients.
    :return: Dictionary of layer names and their corresponding singular values.
    """
    svd_of_gradients = {}
    print(f"computing top {k} singular values of gradients, for each layer")
    # print number of layers
    print(f"number of layers - {len(list(model.named_parameters()))}")
    for name, param in model.named_parameters():
        #print(f"name is {name}")
        if param.requires_grad and param.grad is not None:
            # Flatten the gradient to 2D if it's more than 2 dimensions.
            # SVD can only be applied to matrices (2D tensors).
            # grad = param.grad.data
            # print(f"grad.type - { type(grad)} - grad.shape - {grad.shape}")

            grad = param.grad.detach().cpu().numpy()
            #print(f"grad.type - { type(grad)} - grad.shape - {grad.shape}")
            # print the size of the grad
            if grad.ndim > 2:
                # grad = grad.view(grad.size(0), -1)
                grad = grad.reshape(-1, grad.shape[-1])
            elif grad.ndim == 1:
                grad = grad.reshape(-1, 1)

            print(f"reshaped grad.shape - {grad.shape}")

            try:

                U, s, Vh = randomized_svd(grad, n_components=k, random_state=0)
                svd_of_gradients[name] = s
                print(f"s length - {len(s)}")
            except RuntimeError as e:
                print(f"Error computing SVD for layer {name}: {e}")

    return svd_of_gradients
