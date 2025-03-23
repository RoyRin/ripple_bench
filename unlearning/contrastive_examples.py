


import faiss
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import probes



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
    model_layer_count = probes.get_model_layer_count(model)
    print(f"model_layer_count- {model_layer_count}")
    print(f"init")

    embedding_size = probes.get_embedding_size(model, layer_ind = layer_ind)

    print(f"model embeddinging size {embedding_size}")
    # compute the embedding size 
    search_engine = ImageSearch(model = model, layer_ind = layer_ind, embedding_size=embedding_size)


    N_ = len(probe_dataset)
    for ii, (embedding, embedding_label) in enumerate(zip(probe_dataset, probe_labels)):
        if embedding_label == 0:
            continue
        if ii % (N_//10) == 0:
            print(f"embedding {ii}/{N_}")
        search_engine.add_embedding(embedding = embedding, image_index = ii)
    # add an embedding
    return search_engine
