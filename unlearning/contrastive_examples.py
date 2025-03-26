


import faiss
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from unlearning import probes

from collections import defaultdict
from matplotlib import pyplot as plt

class ImageSearch:
    def __init__(self, model, layer_ind, embedding_size=2048):
        self.model = model 
        self.layer_ind = layer_ind

        self.index = faiss.IndexFlatL2(embedding_size)
        self.image_indices = []

        # Pre-trained ResNet50 for embedding extraction
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def add_embedding(self, embedding, image_index):
        
        #embedding = np.array(embedding).astype('float32').reshape(1, -1)
        embedding = embedding.float().view(1, -1).cpu()
        self.index.add(embedding)
        self.image_indices.append(image_index)


    def search_embedding(self, query_embedding, top_k=5):
        
        #query_embedding = query_embedding.astype('float32').reshape(1, -1)
        query_embedding = query_embedding.float().view(1, -1).cpu()
        distances, indices = self.index.search(query_embedding, top_k)

        results = [(self.image_indices[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results


def build_positive_examples_db(probe_dataset, probe_labels, model, layer_ind):
    model_layer_count = probes.get_layer_count(model)
    print(f"model_layer_count- {model_layer_count}")
    print(f"init")

    embedding_size = probes.get_embedding_size(model, layer_ind = layer_ind)

    print(f"model embeddinging size {embedding_size}")
    # compute the embedding size 
    search_engine = ImageSearch(model = model, layer_ind = layer_ind, embedding_size=embedding_size)


    N_ = len(probe_dataset)
    for ii, (embedding, embedding_label) in enumerate(zip(probe_dataset, probe_labels)):
        if embedding_label == 0: # skip negative labels
            continue
        if ii % (N_//10) == 0:
            print(f"embedding {ii}/{N_}")
        search_engine.add_embedding(embedding = embedding, image_index = ii)
    # add an embedding
    return search_engine

def build_positive_examples_db_from_model(dataset, model, layer_ind, positive_indices, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model_layer_count = probes.get_layer_count(model)
    print(f"model_layer_count- {model_layer_count}")
    print(f"init")

    embedding_size = probes.get_embedding_size(model, layer_ind = layer_ind)

    print(f"model embeddinging size {embedding_size}")
    # compute the embedding size 
    search_engine = ImageSearch(model = model, layer_ind = layer_ind, embedding_size=embedding_size)

    N_ = len(positive_indices)

    embedding_counts = 0    
    for ii, (image_ind) in enumerate(positive_indices):
        embedding_counts += 1

        image, label = dataset[image_ind]

        img_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
        #embedding = get_nth_to_last_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = probes.get_flattened_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = embedding.squeeze(0)
        
        if ii % (N_//10) == 0:
            print(f"embedding {ii}/{N_}; embedding_counts {embedding_counts}")
        search_engine.add_embedding(embedding = embedding, image_index = ii)
    # add an embedding

    return search_engine

def find_closest_examples(search_engine, dataset, neg_indices, layer_ind, model, top_N = 500, N_to_search = 1_000, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    nearest_distances = defaultdict(lambda: 1000000)

    for ii, (index) in enumerate(neg_indices):
        image, label = dataset[index]
        if ii >= N_to_search:
            break

        if ii % (N_to_search//10) == 0:
            print(f"embedding {ii}/{N_to_search}")

        img_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
        #embedding = get_nth_to_last_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = probes.get_flattened_embedding(model.to(device), img_tensor, layer_ind=layer_ind)
        embedding = embedding.squeeze(0)

        #embedding = probes.get_flattened_embedding(model = search_engine.model, image = image.unsqueeze(0), layer_ind = search_engine.layer_ind).squeeze(0)

        results = search_engine.search_embedding(query_embedding = embedding, top_k=5)
        for (ind, dist) in results:
            if ind == ii:
                continue
            nearest_distances[ind] = min(nearest_distances[ind], dist)

    ordered_nearest_distances = sorted(nearest_distances.items(), key=lambda x: x[1])
    print(f"ordered_nearest_distances- len{len(ordered_nearest_distances)}")
    closest_neg_indices = [x[0] for x in ordered_nearest_distances[:top_N]]
    return closest_neg_indices


def create_probe_dataset_random_points(dataset, labels, model, layer_ind, attribute_index, N = 1000):
    """
    Construct a dataset for the probe by randomly selecting all the positive points, and random negative examples from the validation dataset.
    """

    positive_attributes = labels[:, attribute_index] == 1
    negative_attributes = labels[:, attribute_index] == 0
    #
    pos_indices = torch.where(positive_attributes)[0]
    neg_indices = torch.where(negative_attributes)[0]
    #
    N = min(N, len(pos_indices))
    
    # shuffle pos_indices
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)


    ## base probe
    #shuffle neg_indices, choose 1000 random points from neg_indices
    neg_indices_normal = np.random.choice(neg_indices, size=N  , replace=False)


    normal_probe_dataset, normal_probe_dataset_labels = probes.set_up_probe_dataset(model, layer_ind, pos_indices=pos_indices[:N], neg_indices=neg_indices_normal, dataset = dataset, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = True)

    return normal_probe_dataset, normal_probe_dataset_labels






def search_demo(search_engine, probe_dataset, probe_labels, N = 1000):
        # test it out 
    search_engine.search_embedding(query_embedding =  probe_dataset[-1], top_k=5)
    from collections import defaultdict
    nearest_distances = defaultdict(lambda: 1000000)

    for ii, (embedding, embedding_label) in enumerate(zip(probe_dataset, probe_labels)):
        if embedding_label == 1:
            continue
        if ii % (N//10) == 0:
            print(f"embedding {ii}/{N}")
        
        results = search_engine.search_embedding(query_embedding = embedding, top_k=5)
        for (ind, dist) in results:
            if ind == ii:
                continue
            nearest_distances[ind] = min(nearest_distances[ind], dist)

    # plot hist of nearest_distances

    plt.hist(list(nearest_distances.values()), bins=100)
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.title("Histogram of Nearest Distances")
    plt.show()



def create_probe_dataset_contrastive_examples(dataset, labels, model, layer_ind, attribute_index, N = 1000, N_to_search= 2000):
    """
    Construct a dataset for the probe by randomly selecting all the positive points, and the closest negative examples to the positive examples, in the dataset validation dataset.

    N_to_search - number of negative examples to search
    """

    positive_attributes = labels[:, attribute_index] == 1
    negative_attributes = labels[:, attribute_index] == 0
    #
    pos_indices = torch.where(positive_attributes)[0]
    neg_indices = torch.where(negative_attributes)[0]
    #    
    # shuffle pos_indices
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    probe_labels = labels[:, attribute_index] == 1
    print(f"probe_labels {len(probe_labels)} ; {type(probe_labels)}")
    # search the negative examples, for closest examples to the positive examples 
    search_engine = build_positive_examples_db_from_model(dataset=dataset,model= model, positive_indices=pos_indices[:N_to_search], layer_ind = layer_ind)

    print(f"built search engine")
    closest_neg_indices = find_closest_examples(search_engine=search_engine, dataset=dataset, neg_indices = neg_indices[:N_to_search], layer_ind=layer_ind, model= model, top_N = N)
    print(f"len(closest_neg_indices) - {len(closest_neg_indices)}")
    
    # create probe dataset of these closest_neg_attributes, and positive images
    print(f"closest_neg_indices -{len(closest_neg_indices)}")

    closest_probe_dataset, closest_probe_dataset_labels = probes.set_up_probe_dataset(model, layer_ind, pos_indices=pos_indices[:N], neg_indices=closest_neg_indices, dataset = dataset, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = True)
    return closest_probe_dataset, closest_probe_dataset_labels