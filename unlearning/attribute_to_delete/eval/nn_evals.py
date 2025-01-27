from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from numpy import ndarray as NDArray
import numpy as np
import torch


def evaluate_model(model, val_loader):
    is_training = model.training
    model.eval()

    with torch.no_grad():
        total_correct, total_num = 0.0, 0.0
        for ims, labs in tqdm(val_loader):
            ims = ims.cuda()
            labs = labs.cuda()
            with autocast():
                out = model(ims)
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]

    accuracy = total_correct / total_num
    print(f"Accuracy: {accuracy * 100:.1f}%")

    model.train(is_training)
    return accuracy


def get_losses_and_logits(
    model,
    dataloader,
    loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
) -> tuple[NDArray, NDArray]:
    """
    Computes the loss of the model on each point in the dataset.

    Args:
    - model (torch.nn.Module): The model to evaluate.
    - dataloader (torch.utils.data.DataLoader): The dataloader to evaluate the model on.
    - loss_fn (callable, optional): Loss function compatible with the model's output and the dataset's labels.

    Returns:
    - losses (List[float]): List of loss values for each data point in the dataset.
    - logits (List[NDArray]): List of logits for each data point in the dataset.
    """
    # Ensure the model is in evaluation mode
    is_training = model.training
    model.eval()

    # List to store loss values
    losses = []
    logits = []

    # No need to compute gradients
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward pass: compute the model output
            outputs = model(inputs)

            # Compute per-example loss
            loss = loss_fn(outputs, labels)

            losses.extend(loss.detach().cpu().numpy().tolist())
            logits.extend(outputs.detach().cpu().numpy().tolist())

    model.train(is_training)
    return np.array(losses), np.array(logits)
