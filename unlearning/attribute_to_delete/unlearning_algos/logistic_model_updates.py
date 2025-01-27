import numpy as np
from unlearning import utils
from copy import deepcopy
# import logistic regression
from sklearn.linear_model import LogisticRegression as sk_logreg
from unlearning.datasets.utils import subset

def _subset(X, Y, inds_to_exclude):
    mask = np.ones(X.shape[0])
    mask[inds_to_exclude] = 0.
    mask = mask.astype(bool)
    return X[mask], Y[mask]

def model_update_approximation(X, Y, model, forget_inds, algo_name, num_steps=1, penalty= None, C= 100, verbose=False):
    """ wrapper function which calls the appropriate approximation algorithm 
        options:
            - k-LOO
            - LKO
            - exact
        num steps :
            1 or many
    Returns:
        model_weights, 
    """
    # LKO vs k-LOO
    # single-step, multi-step
    approximation_algo = None
    if verbose:
        print(f"algo_name: {algo_name}")
    if algo_name == "k-LOO":
        approximation_algo = model_update_approximation__k_LOO_sum #__safe
    elif algo_name == "LKO":
        approximation_algo = model_update_approximation__woodbury_lko

        #approximation_algo = model_update_approximation__woodbury_lko
    elif algo_name == "exact":
        approximation_algo = model_update_newton_step
    else:
        raise ValueError(f"Unknown approximation algorithm: {algo_name}")

    model_updates = []
    
    model_copy = deepcopy(model)
    model_ = model_copy
    orig_model_weights = model.get_weights()[0]
    for i in range(num_steps):
        model_weights = model_.get_weights()[0]
        
        model_diff = approximation_algo( X, Y, model_, forget_inds, penalty= penalty, C=C, verbose=verbose)

        # if model_diff norm > 100, clip it
        #if np.linalg.norm(model_diff) > 100:
        #    model_diff = model_diff / np.linalg.norm(model_diff) * 100
        
        new_model = model_weights - model_diff
        #print(f"model nrom - {np.linalg.norm(new_model)}")
        # renormalize model
        # new_model = new_model / np.linalg.norm(new_model)

        model_.update_weights(new_model, 0)
        model_updates.append(model_diff)
        
        # model_ = model_ - model_updates
        #print(f"model_ - {model_weights}")
    model_weights = model_.get_weights()[0]#  * np.linalg.norm(orig_model_weights)
    
    update = orig_model_weights - model_weights
    return  update, model_updates



def model_update_newton_step(X,Y,model, forget_inds, max_iter = 1, tol = 1e-10, penalty= None, C = 100, verbose=False, **kwargs):
    # take a newton step on the retain set
    
    orig_model_weights = model.get_weights()[0]
    retain_inds = [i for i in range(len(X)) if i not in forget_inds]
    #X_retain = X[retain_inds]
    #Y_retain = Y[retain_inds]
    X_retain, Y_retain = subset(X, Y, retain_inds)

    sk_logreg_model = sk_logreg(fit_intercept=False, max_iter=max_iter, tol=tol,  solver = 'newton-cg', warm_start=True,  penalty = penalty, C = C)
    sk_logreg_model.coef_ = orig_model_weights
    # take 1 CG step on the retain set
    sk_logreg_model.fit(X_retain, Y_retain)
    # return diff

    # new - original
    update =  orig_model_weights - sk_logreg_model.coef_ 

    return update
####
# LOO
####

def model_update_approximation__k_LOO_sum(X, Y, model, forget_inds, lambda_ = 1e-4,  verbose = False , **kwargs):
    N = len(X)
    print("hi")
    print(f"forget inds - {forget_inds}")
    # cast to 0, 1
    y_0s = ((Y.copy() + 1) / 2).astype(int)
    # returns probability of correct class (0 , 1)
    p = model.predict_proba(X)[np.arange(N), y_0s]
    R = np.diag(p * (1 - p))

    model_update = np.zeros(X.shape[1])
    # print model_update type
    print(f"model_update type - {type(model_update)} - dtype - {model_update.dtype}")

    XRX = X.T @ R @ X
    print(f"model_update type - {type(XRX)} - dtype - {XRX.dtype}")
    # condition number of XRX
    cond_number = np.linalg.cond(XRX)
    print(f"XRX cond - {cond_number}")
    # renormalize X X 
    #XRX = XRX / np.linalg.norm(XRX)
    # normalize X
    X_ = X / np.linalg.norm(X)
    
    XX = X_.T @ X_
    XX = XX / np.linalg.norm(XX)
    cond_number = np.linalg.cond(XX)
    # determinant of XX
    print(f"XX det - {np.linalg.det(XX)}")
    print(f"XX shape - {XX.shape}")
    print(f"xx cond - {cond_number}")
    #print XX [:10, :10]
    print(f"eigen values - {np.linalg.eigvals(XX)[:40]}")
    print(f"XX norm - {np.linalg.norm(XX)}")
    # plot the distribution of the eigenvalues of XRX
    # print eigen vlaues
    #print(np.linalg.eigvals(XX))

    
    #XRX_inv = np.linalg.inv(X.T @ R @ X)
    # Add regularization to the matrix before inversion
    #XRX_inv = np.linalg.pinv(XRX)
    XRX_inv = np.linalg.inv(XRX)
    #XRX_inv = np.linalg.pinv(XRX + lambda_ * np.eye(X.shape[1]))
    print(f"!!! - {XRX_inv.shape}")

    print(f"XRX_inv norm - {np.linalg.norm(XRX_inv)}")
    print(f"forgetinds - {forget_inds} + {len(forget_inds)} + {type(forget_inds)}")
    print("-----")
    for i in forget_inds:
        print(f"forgetting pts {i}")
        leverage = (1 - (X[i] @ XRX_inv @ X[i] * p[i] * (1 - p[i])))
        print(f" computed leverage")
        model_update += Y[i] * XRX_inv @ X[i] * (1 - p[i]) / leverage
        print(f"model update - {model_update}")
    return model_update


def model_update_approximation__k_LOO_sum__safe(X, Y, model, forget_inds, lambda_ = 1e-4,  verbose = False ):
    N = len(X)
    # cast to 0, 1
    y_0s = ((Y.copy() + 1) / 2).astype(int)
    # returns probability of correct class (0 , 1)
    p = model.predict_proba(X)[np.arange(N), y_0s]
    R = np.diag(p * (1 - p))

    model_update = np.zeros(X.shape[1])
    XRX = X.T @ R @ X
    #XRX_inv = np.linalg.inv(X.T @ R @ X)
    # Add regularization to the matrix before inversion
    cond_number = np.linalg.cond(XRX)
    print(f"XRX cond - {cond_number}")
    lambda_ = lambda_ * 100
    XRX_inv = np.linalg.pinv(XRX + lambda_ * np.eye(X.shape[1]))

    print(f"XRX_inv norm - {np.linalg.norm(XRX_inv)}")
    for i in forget_inds:
    
        leverage = (1 - (X[i] @ XRX_inv @ X[i] * p[i] * (1 - p[i])))
        # Clipping the leverage to avoid division by very small numbers

        leverage = max(leverage, 1e-4)
        model_update += Y[i] * XRX_inv @ X[i] * (1 - p[i]) / leverage
    return model_update

        
    
####
# LKO
####


def woodbury_approximation(A_inv, U, C, V):
    """
    Compute the inverse of A + UCV using the Woodbury matrix identity.

    Parameters:
    A : numpy.ndarray
        An invertible square matrix.
    U, V : numpy.ndarray
        Matrices such that the product UCV is well-defined and matches the dimensions of A.
    C : numpy.ndarray
        A square matrix.

    Returns:
    numpy.ndarray
        The inverse of A + UCV computed using the Woodbury matrix identity.
    """
    middle_term_inv = np.linalg.inv(C + V @ A_inv @ U)
    woodbury_inv = A_inv - (A_inv @ U @ middle_term_inv @ V @ A_inv)
    return woodbury_inv



def model_update_approximation__woodbury_lko(X, Y, 
                                             model,
                                             forget_inds,
                                             verbose=False, **kwargs):
    """
    returns equations 26b analogue from the paper
        \theta* - \theta*_{-k} = - (X^T R X)_{-k}^{-1} X_{-k}^T q_{-k} 

    Compute the influence for a given data point in logistic regression.
    Note: to compute    `f(z; θ(S)) - f(z; θ(S \ zi))`
        need to take cross product of `z` and return statement

    :param X: The matrix of all input features (n x k-dimensional).
    forget_inds : indices of the data points to forget
    return :
        influence of ith data point 
    """
    retain_inds = [i for i in range(len(X)) if i not in forget_inds]

    #
    # OBSERVE! - y_0s are ignored here. Surprisingly this improves accuracy.
    #
    y_0s = ((Y.copy() + 1) / 2).astype(int)
    predictions = model.predict_proba(X)[np.arange(len(X)), y_0s]
    #predictions = model.predict_proba(X)[np.arange(len(X)), 0]
    #predictions_forget = predictions[forget_inds]
    predictions_retain = predictions[retain_inds]

    
    R = predictions * (1 - predictions)

    XRX = X.T @ np.diag(R) @ X

    # UCV_2 = X_forget.T @ np.diag(R[forget_inds]) @ X_forget
    
    X_forget = X[forget_inds]
    X_retain = X[retain_inds]
    
    #U = X_forget.T
    #C = np.diag(R[forget_inds])
    # V = X_forget
    U = X_retain.T
    C = np.diag(R[retain_inds])
    V = X_retain

    XRX_inv = np.linalg.pinv(XRX)
    A_inv = XRX_inv
    A_inv_LKO = woodbury_approximation(A_inv, U, C, V)

    #theta_lko_approx_delta = A_inv_LKO @ X_forget.T
    #theta_lko_approx_delta = np.dot(theta_lko_approx_delta, predictions_forget)
    theta_lko_approx_delta = A_inv_LKO @ X_retain.T
    theta_lko_approx_delta = np.dot(theta_lko_approx_delta, predictions_retain)
    
    return theta_lko_approx_delta




####

################################################################################
################################################################################
################################################################################

####
# Extract influence estimate from modeul update approximation
####

def get_influence_estimate(x, X, model, forget_inds, approximation_algo, verbose=False):
    """returns the leave one out estimate for a given data point
        f(x; θ(S)) - f(x; θ(S \ forget_i)

    Args:
        "approximation_algo" : follows interface:
            f(X, model, forget_inds, verbose=False)

    Returns:
        _type_: _description_
    """
    # model_diff__loo_approximation = model_update_approximation__sherman_morrison_loo( X, model, forget_i, verbose=verbose)

    model_diff_approximation = approximation_algo(
        X, model, forget_inds, verbose=verbose)
    # x_ = x.reshape(1, -1)
    LOO_infl = x @ model_diff_approximation
    prediction = model.predict_proba(x)[0][0]
    estimate = utils.invert_logistic(prediction) - LOO_infl
    return estimate


####
###
###
