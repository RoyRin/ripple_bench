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
import datetime

# check if model exists
from pathlib import Path
CWD = Path.cwd()

from pathlib import Path
BATCH_SIZE = 64  # Increase batch size for better training
NUM_WORKERS = 4
DATA_DIR = "data"

BASE_DIR = Path("/n/home04/rrinberg/code/data_to_concept_unlearning")

DATA_DIR= BASE_DIR/ Path("/notebooks/data")
DATA_DIR = Path("/n/home04/rrinberg/data_dir__sneel/Lab/rrinberg/")
DATA_DIR = Path("/n/home04/rrinberg/data_dir/data_to_concept/")

SAVE_DIR = BASE_DIR/ Path("notebooks/models")





# given a model, look at the embedding layer, and see how much a concept is present in the model by :
# take training data with labels of 1/0 for the concept
# pass the data through the model
# take the embeddings from the embedding layer
# then see if the embeddings are separable by the concept, using a logistic regression

import pickle 
def save_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

import torch
import torch.nn as nn

def get_nth_to_last_embedding(model, x, n=1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Extract embedding from the n-th to last layer of ResNet.
    Args:
        model: ResNet model (e.g., ResNet50)
        x: Input image tensor (batch_size, 3, H, W)
        n: Number of layers to remove from the end (default: 2 for second-to-last layer)
    Returns:
        features: Extracted feature tensor
    """
    
    x = x.to(device )
    model = model.to(device)
    # Get model layers except last n layers
    truncated_model = nn.Sequential(*list(model.children())[:-n])
    with torch.no_grad():
        features = truncated_model(x)  # Pass input through the truncated model
        features = features.squeeze(-1).squeeze(-1)  # Remove unnecessary dimensions

    return features

# Example usage:
# x = torch.randn(1, 3, 224, 224).to(device)  # Example input
# embedding = get_nth_to_last_embedding(model, x, n=2)  # Extract second-to-last layer
# print(embedding.shape)

def get_embedding(model, x, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return get_nth_to_last_embedding(model, x, n=1, device = device )




def create_embeddding_dataset(model_, train_dataset, indices_with_attr, indices_without_attr, num_images = 1000, DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")):


    # Get embeddings for images without attribute
    embeddings_without_attr = []
    for ii, idx in enumerate(indices_without_attr[:num_images]):
        if ii % (num_images//2) == 0:
            print("Processing sample", ii)
        img_tensor = train_dataset[idx][0].unsqueeze(0)
        embedding = get_embedding(model_, img_tensor, device = DEVICE)
        embeddings_without_attr.append(embedding.cpu().numpy())
    print("-----2-----")
    
    
    # Get embeddings for images with attribute
    embeddings_with_attr = []

    for ii, idx in enumerate(indices_with_attr[:num_images]):
        if ii % (num_images//2) == 0:
            print("Processing sample", ii)
        img_tensor = train_dataset[idx][0].unsqueeze(0)  
        embedding = get_embedding(model_, img_tensor, device = DEVICE)
        embeddings_with_attr.append(embedding.cpu().numpy())

    all_embeddings = embeddings_without_attr + embeddings_with_attr
    
    # stack them together 
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    labels_ = np.concatenate([np.zeros(len(embeddings_without_attr)), np.ones(len(embeddings_with_attr))])

    return all_embeddings, labels_






def load_model(attr_index, models_dir, DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    SAVE_MODEL_PATH = f"resnet50_celeba__remove_{attr_index}.pth"
    model_path = models_dir / SAVE_MODEL_PATH
    if not model_path.exists():
        raise ValueError(f"Model not found at {model_path}")
    
    model_without_attr = models.resnet50(pretrained=False)  # Train from scratch
    num_ftrs = model_without_attr.fc.in_features
    model_without_attr.fc = nn.Linear(num_ftrs, 40)  # Modify final layer for 40 attributes
    model_without_attr = model_without_attr.to(DEVICE)

    model_without_attr.load_state_dict(torch.load(model_path))
    
    return model_without_attr

if __name__ == "__main__":



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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================
    # 8. TRAIN & EVALUATE MODEL
    # ========================
    train_labels = train_dataset.attr
    val_labels = val_dataset.attr

    # print the labels names
    for i, name in enumerate(train_dataset.attr_names):
        print(f"{i}: {name}") 


    models_dir = BASE_DIR / "notebooks" / "models"
    models_dir = BASE_DIR / "notebooks" / "models_without_labels"
    
    PLOT_DIR =models_dir / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    min_ = 0 # -1 #HACK TODO - add in the "trained on full dataset model"
    max_ = 40
    results = {}
    
    start_time = datetime.datetime.now()
    for attr_index__model in range(min_, max_): # model was not trained 
        model_round_start = datetime.datetime.now()
        print(f"total time elapsed: {model_round_start - start_time}")
        for attr_index__to_evaluate_on in range(0, max_):  # concept to evaluate on
            round_start_time = datetime.datetime.now()
            print(f"attr_index__dataset - {attr_index__to_evaluate_on}")
            print(f"attr_index__model - {attr_index__model}")
            if False:
                attr_index__to_evaluate_on = 4
                attr_index__model = 3
            
            

            
            model = load_model(attr_index__model, models_dir)

            val_indices_without_attr = np.where(val_labels[:, attr_index__to_evaluate_on] == 0)[0]
            

            val_indices_with_attr = np.where(val_labels[:, attr_index__to_evaluate_on] == 1)[0]

            # shuffle val_indices_without_attr
            np.random.shuffle(val_indices_without_attr)
            np.random.shuffle(val_indices_with_attr)

            print(f"first 10 indices with attr {val_indices_with_attr[:10]}")
            print(f"first 10 indices without attr {val_indices_without_attr[:10]}")

            # CREATE embedding layer dataset
            num_images=  1000
            all_embeddings, labels_ = create_embeddding_dataset(model, val_dataset, val_indices_with_attr, val_indices_without_attr, num_images = num_images, DEVICE = device)
            # print shapes
            print("all_embeddings shape", all_embeddings.shape)
            print("labels shape", labels_.shape)

            ### train logistic regression on embeddings and labels

            # train logistic regression on embeddings and labels
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split
            # split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(all_embeddings, labels_, test_size=0.2, random_state=42)
            # train logistic regression
            logreg = LogisticRegression(max_iter=1000, solver="liblinear")
            logreg.fit(X_train, y_train)
            preds = logreg.predict(X_test)
            acc = accuracy_score(y_test, preds)

            results[(attr_index__model, attr_index__to_evaluate_on)] = acc

            #acc = accuracy_score(labels_, preds)
            print(f"Accuracy of logistic regression on embeddings for {attr_index__to_evaluate_on}: {acc:.4f}")
            print(f"embedding shape {all_embeddings.shape}")
            # run PCA on embeddings
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import numpy as np
            #embeddings_without_attr_np = np.concatenate(embeddings_without_attr, axis=0)
            #embeddings_with_attr_np = np.concatenate(embeddings_with_attr, axis=0)
            #embeddings_with_attr_np.shape
            # plot embeddings with PCA
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(all_embeddings)

            plt.figure(figsize=(10, 8))
            plt.scatter(pca_embeddings[:num_images, 0], pca_embeddings[:num_images, 1], label="Image Without Attribute", alpha=0.5)
            plt.scatter(pca_embeddings[num_images:, 0], pca_embeddings[num_images:, 1], label="Image With Attribute", alpha=0.5)
            plt.title(f"PCA of Embeddings: Accuracy: {acc:.4f} \n(Model excluding {attr_index__model}); (Data evaluated on {attr_index__to_evaluate_on}) ")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.legend()
            plt.tight_layout()
            save_path = PLOT_DIR / f"PCA_embeddings__model_{attr_index__model}__data_{attr_index__to_evaluate_on}"    
            print("saving plot to ", save_path)
            plt.savefig(f"{save_path}.pdf")
            plt.savefig(f"{save_path}.png")

            #plt.show()
            
            # print times
            round_end_time = datetime.datetime.now()
            print(f"round time elapsed: {round_end_time - round_start_time}")
            print(f"total time elapsed: {round_end_time - start_time}")
            print(f"model time elapsed: {round_end_time - model_round_start}")
            # save results for eahc of the 1500
            

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # save data:
    fn = f"evals_results__{date_str}.pkl"
    save_pickle(results, models_dir / fn)
    print("saved results to ", models_dir / fn)