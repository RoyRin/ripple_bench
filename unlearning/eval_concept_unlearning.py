# given a model, look at the embedding layer, and see how much a concept is present in the model by :
# take training data with labels of 1/0 for the concept
# pass the data through the model
# take the embeddings from the embedding layer
# then see if the embeddings are separable by the concept, using a logistic regression
