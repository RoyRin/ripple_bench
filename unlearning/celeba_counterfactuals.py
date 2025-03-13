import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np 

from torch.utils.data import DataLoader, Subset
import json
import yaml  
def save_yaml(params, filename):
    with open(filename, 'w') as file:
        yaml.dump(params, file)

import pickle 
def save_pickle(params, filename):
    with open(filename, 'wb') as file:
        pickle.dump(params, file)
from pathlib import Path    

BATCH_SIZE = 64  # Increase batch size for better training
NUM_WORKERS = 4
DATA_DIR = "data"
DATA_DIR= Path("/n/home04/rrinberg/code/data_to_concept_unlearning/notebooks/data")
DATA_DIR = Path("/n/home04/rrinberg/data_dir__sneel/Lab/rrinberg/")
DATA_DIR = Path("/n/home04/rrinberg/data_dir/data_to_concept/")

BASE_DIR = Path("/n/home04/rrinberg/code/data_to_concept_unlearning/notebooks/")





DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MPS device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
print("Using device:", DEVICE)




# ========================
# 5. TRAINING FUNCTION
# ========================
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, SAVE_MODEL_PATH= None, DEVICE = "cuda"):
    val_losses, val_accuracies, per_class_accuracies = [], [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(DEVICE), labels.float().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        # Validation
        val_loss = validate(model, val_loader, criterion)
        per_class_accuracy, overall_accuracy = evaluate(model, val_loader)
        scheduler.step()  # Update learning rate
        val_loss = float(val_loss)
        overall_accuracy = float(overall_accuracy)
        per_class_accuracy = [float(x) for x in per_class_accuracy]
        
        val_losses.append(val_loss)
        val_accuracies.append(overall_accuracy)
        per_class_accuracies.append(per_class_accuracy)

        if SAVE_MODEL_PATH is not None:
            print(f"Saving model to {SAVE_MODEL_PATH}")
            # Save model checkpoint
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Training complete.")
    return val_losses, val_accuracies, per_class_accuracies
# ========================
# 6. VALIDATION FUNCTION
# ========================
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.float().to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

# ========================
# 7. EVALUATION FUNCTION
# ========================
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.float().to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)  # Convert logits to probabilities
            preds = (probs > 0.5).int()  # Threshold at 0.5 to get binary predictions

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    preds = torch.cat(all_preds)
    true_labels = torch.cat(all_labels)

    # Compute per-class accuracy
    per_class_accuracy = (preds == true_labels).float().mean(dim=0)

    # Compute overall accuracy (average across all attributes)
    overall_accuracy = per_class_accuracy.mean().item()

    # Print results
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Per-attribute accuracy:\n{per_class_accuracy}")

    return per_class_accuracy, overall_accuracy




if __name__ == "__main__":
    
    import sys 
    index = int(sys.argv[1])
    # HACK to be able to get "train on full dataset"
    index = index -1

    

    TRIALS = 3

    train_without_data = False  # train without data that has 1 attribute
    train_without_data = True 
    train_without_labels = not train_without_data # train on full dataset, but 0-out labels for attribute
    if train_without_data:
        SAVE_DIR = BASE_DIR / "models_without_data__multiple"
    else:
        SAVE_DIR = BASE_DIR / "models_without_labels__multiple"
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ========================
    # 2. DATA LOADING (WITH STRONG AUGMENTATION)
    # ========================
    transform = transforms.Compose([
        #transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),  # Augmentation: Flip images randomly
        transforms.RandomRotation(10),  # Augmentation: Small rotations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jittering
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = CelebA(root=DATA_DIR, split="train", transform=transform, download=True, target_type="attr")
    val_dataset = CelebA(root=DATA_DIR, split="valid", transform=transform, download=True, target_type="attr")

        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    

    # ========================
    # 1. CONFIGURATION
    # ========================
    #IMAGE_SIZE = 224
    NUM_EPOCHS = 20  # More epochs needed
    LEARNING_RATE = 0.01  # Higher initial LR when training from scratch


    # ========================
    # 8. TRAIN & EVALUATE MODEL
    # ========================
    train_labels = train_dataset.attr
    val_labels = val_dataset.attr

    ####
    ### TODO - parse out the attribute to remove
    ###
    print(f"index - {index}")
    attribute_ = ""
    attr_index = index
    if index != -1:
        if train_without_data:
            print(f"train on dataset without data of attribute {attr_index}")
            attribute_ = train_dataset.attr_names[attr_index]

            indices_without_attr = np.where(train_labels[:, attr_index] == 0)[0]
            print("Number of samples without attribute:", len(indices_without_attr))
            dataset_without_attribute = Subset(train_dataset, indices_without_attr)
            len(dataset_without_attribute)
            # train on dataset without attribute
            print("Attribute to remove:", attribute_)
        else:
            print("train on full dataset, but zero out labels for 1 attribute")
            # train without labels, so zero out labels for 1 attribute attr_index
            train_dataset.attr[:, attr_index] = 0 
            dataset_without_attribute = train_dataset
            
    else:
        print("train on full dataset")
        dataset_without_attribute = train_dataset

    train_loader_without_attr = DataLoader(dataset_without_attribute, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)



    
    print(f"original size: {len(train_dataset)}")
    print(f"new size: {len(dataset_without_attribute)}")
    
    for trial in range(TRIALS):
        print(f"Trial {trial+1}/{TRIALS}")


        ####
        # ========================
        # 3. MODEL DEFINITION (TRAINING FROM SCRATCH)
        # ========================
        model = models.resnet50(pretrained=False)  # Train from scratch
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 40)  # Modify final layer for 40 attributes
        model = model.to(DEVICE)

        # ========================
        # 4. LOSS & OPTIMIZER
        # ========================
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)  # SGD with momentum

        # Learning rate scheduler (reduce LR after plateaus)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)  # Reduce LR every 15 epochs

        ####
        if attr_index != -1:
            SAVE_MODEL_PATH = SAVE_DIR / f"resnet50_celeba__remove__{attr_index}__{trial}.pth"
        else:
            SAVE_MODEL_PATH = SAVE_DIR / f"resnet50_celeba__full__{trial}.pth"

        val_losses, val_accuracies, per_class_accuracies = train(model, train_loader_without_attr, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, SAVE_MODEL_PATH=SAVE_MODEL_PATH)
        
        #final_eval = evaluate(model, val_loader)
        #val_losses = [round(x, 4) for x in val_losses]
        #val_accuracies = [round(x, 4) for x in val_accuracies]
        #per_class_accuracies = [round(x, 4) for x in per_class_accuracies]

        results = {
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "per_class_accuracies": per_class_accuracies,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "SGD",
            "scheduler": "StepLR",
            "attribute": attribute_,
            "attribute_index": attr_index,
            "model": "resnet50",
        }
        save_yaml(results, SAVE_DIR/ f"results_{attr_index}.yaml")

        save_pickle(results, SAVE_DIR/ f"results_{attr_index}.pkl")
