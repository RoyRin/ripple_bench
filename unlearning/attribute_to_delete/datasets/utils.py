import numpy as np

def _subset(X, Y, inds_to_exclude):
    mask = np.ones(X.shape[0])
    mask[inds_to_exclude] = 0.
    mask = mask.astype(bool)
    return X[mask], Y[mask]


def subset(X, Y, inds_to_include):
    mask = np.zeros(X.shape[0])
    mask[inds_to_include] = 1.
    mask = mask.astype(bool)
    return X[mask], Y[mask]

def add_random_points_to_dataset(X, Y, num_points ):
    """ 
    
    """
    # add random points to the dataset, with random labels
    n,d = X.shape
    labels_set = np.unique(Y)
    new_X = np.random.rand(num_points,d)
    new_Y = np.random.choice(labels_set, num_points)
    X = np.concatenate((X,new_X))
    Y = np.concatenate((Y,new_Y))
    return X,Y


def add_random_points_to_dataset_FIT(X, Y, num_points , extra_dim = 1, verbose = False):
    """ 
    
    """
    # original shapes
    if verbose:
        print(f"X.shape - {X.shape}")
        print(f"Y.shape - {Y.shape}")
    if num_points > X.shape[0]:
        raise ValueError(f"num_points: {num_points} is greater than the number of points in X: {X.shape[0]}")
    # add random points to the dataset, with random labels
    n,d = X.shape
    labels_set = np.unique(Y)

    # add 1 extra column to X, of all 0s
    new_X = np.hstack((X, np.zeros((n, extra_dim))))
    # create `extra_dim` stacks of the Y labels
    stacked_Y = np.tile(Y, (extra_dim,1)).T
    if verbose:
        print(f"stacked_Y.shape - {stacked_Y.shape}")
    
    #FIT_addition = np.hstack((X, np.ones((n, extra_dim))  ))
    FIT_addition = np.hstack((X, stacked_Y ))
    if verbose:
        print(f"FIT_addition.shape - {FIT_addition.shape}")
    FIT_addition = FIT_addition[:num_points,:]
    #extra_X = np.random.rand(num_points,d+ extra_dim)
    #extra_X = np.hstack((X[:num_points], np.ones((num_points, extra_dim))))
    if verbose:    
        #extra_Y = np.random.choice(labels_set, num_points)
        print(f"FIT_addition.shape 0 {FIT_addition.shape}")
        print(f"FIT_addition.shape 0 {FIT_addition[:num_points,:].shape}")
        print(f"np.concatenate((Y,Y[:num_points])) .shape 0 {np.concatenate((Y,Y[:num_points])).shape}")
        
    X = np.concatenate((new_X, FIT_addition[:num_points]))
    Y = np.concatenate((Y,Y[:num_points]))

    return X,Y
