import torch 
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
import torchvision.datasets as datasets
from pathlib import Path

BASE_DIR = Path("/n/home04/rrinberg/data_dir__sneel/Lab/rrinberg/results")


def load_model(attr_index, models_dir, trial = 1, DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if attr_index is None:
        SAVE_MODEL_PATH = f"resnet50_celeba.pth"
    else:
        SAVE_MODEL_PATH = f"resnet50_celeba__remove__{attr_index}__{trial}.pth"
    # resnet50_celeba__remove__31__0
    model_path = models_dir / SAVE_MODEL_PATH
    if not model_path.exists():
        raise ValueError(f"Model not found at {model_path}")
    
    model_without_attr = models.resnet50(pretrained=False)  # Train from scratch
    num_ftrs = model_without_attr.fc.in_features
    model_without_attr.fc = nn.Linear(num_ftrs, 40)  # Modify final layer for 40 attributes
    model_without_attr = model_without_attr.to(DEVICE)

    model_without_attr.load_state_dict(torch.load(model_path))
    
    return model_without_attr


# validate the model on val set
from tqdm import tqdm   
def evaluate(model, dataloader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), max_rounds = None):
    model.eval()
    all_preds, all_labels = [], []
    steps = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            if max_rounds is not None and steps >= max_rounds:
                break
            steps+=1

            images, labels = images.to(device), labels.float().to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)  # Convert logits to probabilities
            preds = (probs > 0.5).int()  # Threshold at 0.5 to get binary predictions

            all_preds.append(preds)
            all_labels.append(labels)

    # Concatenate all predictions and labels
    preds = torch.cat(all_preds).cpu()
    labels = torch.cat(all_labels).cpu()
    return preds, labels


def get_TP_TN_FP_FN(col_ind, labels, predictions):
    # Get the false positives and false negatives for a specific attribute
    # where labels[:, col_ind] == 0 and predictions[:, col_ind] == 1
    FP = ((labels[:, col_ind] == 0) & (predictions[:, col_ind] == 1)).sum()
    FN = ((labels[:, col_ind] == 1) & (predictions[:, col_ind] == 0)).sum()
    TP = ((labels[:, col_ind] == 1) & (predictions[:, col_ind] == 1)).sum()
    TN = ((labels[:, col_ind] == 0) & (predictions[:, col_ind] == 0)).sum()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0 
    return (TP, TN, FP, FN), (precision, recall)



