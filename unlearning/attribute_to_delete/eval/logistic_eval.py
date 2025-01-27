import numpy as np
from scipy.stats import pearsonr, spearmanr

from unlearning.datasets import utils
from unlearning.datasets.utils import subset
from unlearning.models import logistic_regression
from unlearning.models.logistic_regression import fit_log_reg
from unlearning.unlearning_algos import logistic_model_updates


def evaluate_model_update(datasets, original_model, retrained_model, forget_inds, algo_name, newton_steps=1, verbose = False, penalty= None, C= 100 ):  
    (X_train, Y_train, X_test, Y_test) = datasets
    N = X_train.shape[0]
    retain_inds = np.array([i for i in range(N) if i not in forget_inds])
    X_retain, Y_retain = subset(X_train, Y_train, retain_inds)
    X_forget, Y_forget = subset(X_train, Y_train, forget_inds)
    

    if algo_name == "lko":
        algo = logistic_model_updates.model_update_approximation__woodbury_lko
    elif algo_name == "k_loo":
        algo = logistic_model_updates.model_update_approximation__k_LOO_sum
    elif algo_name == "newton":
        algo = logistic_model_updates.model_update_newton_step
    else:
        raise ValueError(f"unknown algo {algo_name}")
    # model update returns a DIFF from the original model
    model_update = algo(X_train, Y_train, original_model, forget_inds, max_iter=newton_steps, penalty=penalty, C=C,  verbose=verbose) # 

    original_weights = original_model.get_weights()[0]
    retrained_weights = retrained_model.get_weights()[0]
    approx_weights = original_weights - model_update
    #
    # compute model diifs 
    #     
    l2_diffs,  l2_diffs_normalized, update_angle_, model_angle_, spearman_val = report_difference_between_models(original_weights, retrained_weights, approx_weights, verbose = False )

    model_ = logistic_regression.LogisticRegression_wrapper(bias = False, max_iter=3000 ,tol=1e-6)
    # fit just to get the right dims
    model_.fit(X_forget, Y_forget)
    model_.fit(X_train, Y_train)
    model_.update_weights(approx_weights, 0)
    
    retain_accuracy, forget_accuracy, val_accuracy = evaluate_model(model_, X_retain, Y_retain, X_forget, Y_forget, X_test, Y_test)
    
    # get FIT results
    FIT_test_results = None
    try:
        FIT_test_results = FIT_test(X_train, Y_train, algo_name, newton_steps =newton_steps , penalty= penalty, C= C, verbose = verbose)
        #print(f"penalty for fit test - {penalty}")
    except Exception as e:
        print(f"FIT test failed: {e}")
        pass
    results = {
        #"l2_diffs": l2_diffs,
        #"l2_diffs_normalized": l2_diffs_normalized,
        "update_angle": update_angle_,
        "model_angle": model_angle_,
        "spearman": spearman_val.statistic,
        "retain_accuracy": retain_accuracy,
        "forget_accuracy": forget_accuracy,
        "val_accuracy": val_accuracy,
        "FIT_test_results": FIT_test_results,
    }
    return results

    if verbose:
        # print the weights, and the new wegihts
        print(f"original weights - {original_weights / np.linalg.norm(original_weights)}")
        print(f"retrained weights - {retrained_weights / np.linalg.norm(retrained_weights)}")
        new_model_norm = (original_weights - model_update) / np.linalg.norm(original_weights - model_update)
        print(f"new model - {new_model_norm}")
        l2_diffs,  l2_diffs_normalized, update_angle, model_angle, spearman_val = report_difference_between_models(original_weights, retrained_weights, original_weights - model_update, verbose = False )


def FIT_test(X_train, Y_train, algo_name, newton_steps =1 , penalty= None, C= 100, num_points = 4, verbose = False):
    """
    take X , Y 
    add in an extra dimension to X, 

    take dataset X of dim n x d
    add a column of all 0's, so it's n x (d+1)
    add in a few points where the d+1 -th col is non-zero.
    learned logistic regression will have non-zero values for the d+1-th column
    we can evaluate unlearning based on how close to 0 the d+1-th column is

    Note: this isn't the most efficient use of models, and will retrain 3 models each time is run. if this becomes expensive, 2 of the models could be passed in (original and retrained)
    """ 

    
    X_added, Y_added  = utils.add_random_points_to_dataset_FIT(X_train, Y_train,
                                                        num_points )
    forget_inds = list(range(X_train.shape[0], X_added.shape[0]))
    
    retain_X = X_added[ : -num_points]
    retain_Y = Y_added[: -num_points]
    forget_X = X_added[ -num_points:]
    forget_Y = Y_added[-num_points:]
    
    original_model = fit_log_reg(X_added, Y_added,verbose = False )
    retrained_model =  fit_log_reg(retain_X, retain_Y,verbose = False )
    # get the model update
    if algo_name == "lko":
        algo = logistic_model_updates.model_update_approximation__woodbury_lko
    elif algo_name == "k_loo":
        algo = logistic_model_updates.model_update_approximation__k_LOO_sum
    elif algo_name == "newton":
        algo = logistic_model_updates.model_update_newton_step
    else:
        raise ValueError(f"unknown algo {algo_name}")

    model_update = algo(X_added, Y_added, original_model, forget_inds, max_iter=newton_steps, penalty=penalty, C=C,  verbose=verbose) # 

    original_weights = original_model.get_weights()[0]
    retrained_weights = retrained_model.get_weights()[0]
    approximated_weights = (original_weights - model_update) 

    original_FIT = original_weights.flatten()[-1]
    retrained_FIT = retrained_weights.flatten()[-1]
    model_update_FIT = approximated_weights.flatten()[-1]

    
    if verbose:
        # eval 3 models on retain and forget sets 
        print(f"original model retain acc : {original_model.evaluate_model(retain_X, retain_Y)}")
        print(f"retrained model retain acc : {retrained_model.evaluate_model(retain_X, retain_Y)}")
        print(f"original model forget acc : {original_model.evaluate_model(forget_X, forget_Y)}")
        print(f"retrained model forget acc : {retrained_model.evaluate_model(forget_X, forget_Y)}")
        
        updated_model = logistic_regression.LogisticRegression_wrapper(bias = False, max_iter=3000 ,tol=1e-6)
        updated_model.fit(retain_X, retain_Y)
        #updated_model.update_weights(original_weights, 0)
        updated_model.update_weights(approximated_weights, 0)

        print(f"updated model retain acc : {updated_model.evaluate_model(retain_X, retain_Y)}")
        print(f"updated model forget acc : {updated_model.evaluate_model(forget_X, forget_Y)}")

    # beep boop baap - need to figure out if this is correctly implemented.
    # are the model updates correct? 

    #return original_model, retrained_model, model_update
    return (original_FIT, retrained_FIT, model_update_FIT)


def evaluate_model(model, retain_X, retain_Y, forget_X, forget_Y, val_X, val_Y):
    # get accuracy on each set:
    #print(f"shape - retain_X: {retain_X.shape}, retain_Y: {retain_Y.shape}")
    retain_accuracy = model.evaluate_model(retain_X, retain_Y)
    #print(f"shape - forget_X: {forget_X.shape}, forget_Y: {forget_Y.shape}")
    forget_accuracy = model.evaluate_model(forget_X, forget_Y)
    #print(f"shape - val_X: {val_X.shape}, val_Y: {val_Y.shape}")
    val_accuracy = None
    if val_X is not None:
        val_accuracy = model.evaluate_model(val_X, val_Y)

    return retain_accuracy, forget_accuracy, val_accuracy


def l2_difference(original_weights, retrained_weights, approximation_weights):
    """ calculate the L2 difference between the models

    Args:
        original_weights (np.array): weights of the original model
        retrained_weights (np.array): weights of the retrained model
        approximation_weights (np.array): weights of the approximation model

    Returns:
        float: L2 difference between the models
    """
    orig_to_retrained = (original_weights - retrained_weights)
    orig_to_approx = (original_weights - approximation_weights)
    retrained_to_approx = (retrained_weights - approximation_weights)

    orig_to_retrained_diff = np.linalg.norm(orig_to_retrained)
    orig_to_approx_diff = np.linalg.norm(orig_to_approx)
    retrained_to_approx_diff = np.linalg.norm(retrained_to_approx)
    return orig_to_retrained_diff, orig_to_approx_diff, retrained_to_approx_diff

def l2_difference_normalized(original_weights, retrained_weights, approximation_weights):
    orig_to_retrained_diff, orig_to_approx_diff, retrained_to_approx_diff = l2_difference(original_weights, retrained_weights, approximation_weights)
    normalizer = np.linalg.norm(original_weights)

    return orig_to_retrained_diff / normalizer, orig_to_approx_diff / normalizer, retrained_to_approx_diff / normalizer

def update_angle(original_weights, retrained_weights, approximation_weights):
    """ calculate the angle between the models

    Args:
        original_weights (np.array): weights of the original model
        retrained_weights (np.array): weights of the retrained model
        approximation_weights (np.array): weights of the approximation model

    Returns:
        float: angle between the models
    """
    orig_to_retrained = (original_weights - retrained_weights)
    orig_to_approx = (original_weights - approximation_weights)

    retrained_to_approx = (retrained_weights - approximation_weights)

    dot_product = np.dot(orig_to_retrained, orig_to_approx.T)
    LKO_to_retrained_angle = np.degrees(np.arccos(
        dot_product / (np.linalg.norm(orig_to_retrained) * np.linalg.norm(orig_to_approx))))

    return LKO_to_retrained_angle


def model_angle(retrained_weights, approximation_weights):
    """ calculate the angle between the models

    Args:
        original_weights (np.array): weights of the original model
        retrained_weights (np.array): weights of the retrained model
        approximation_weights (np.array): weights of the approximation model

    Returns:
        float: angle between the models
    """

    dot_product = np.dot(retrained_weights, approximation_weights.T)
    angle = np.degrees(np.arccos(
        dot_product / (np.linalg.norm(retrained_weights) * np.linalg.norm(approximation_weights))))

    return angle

def spearman_correlation(original_weights, retrained_weights, approximation_weights):
    """ test the spearman correlation"""
    original_to_retrained = (original_weights - retrained_weights).reshape(-1)
    original_to_approx = (original_weights - approximation_weights).reshape(-1)
    #print(f"original_to_retrained: {original_to_retrained}")
    #print(f"original_to_approx: {original_to_approx}")
    spearman_val= spearmanr(original_to_retrained, original_to_approx)
    return spearman_val    

def report_difference_between_models(original_weights, retrained_weights, approximation_weights, verbose = False):
    """ print out the difference between the models

    Args:
        original (_type_): _description_
        retrained (_type_): _description_
        retrained_approximation (_type_): _description_
    """
    # original_weights = original.coef_
    # retrained_weights = retrained.coef_
    # approximation_weights = retrained_approximation.coef_
    orig_to_retrained_diff, orig_to_approx_diff, retrained_to_approx_diff = l2_difference(original_weights, retrained_weights, approximation_weights)
    l2_diffs = orig_to_retrained_diff, orig_to_approx_diff, retrained_to_approx_diff
    
    orig_to_retrained_normalized, orig_to_approx_normalized, retrained_to_approx_normalized = l2_difference_normalized(original_weights, retrained_weights, approximation_weights)
    l2_diffs_normalized = orig_to_retrained_normalized, orig_to_approx_normalized, retrained_to_approx_normalized

    update_angle_ = update_angle(original_weights, retrained_weights, approximation_weights)

    model_angle_ = model_angle(retrained_weights, approximation_weights)

    spearman_val = spearman_correlation(original_weights, retrained_weights, approximation_weights)
    if verbose:
        print(f"---- L2 Difference -----")
        print(f"Orig to Retrained (normalized): {orig_to_retrained_normalized}")
        print(f"Orig to retrained approx (normalized): {orig_to_approx_normalized}")
        print(f"Retrained to retrained-approx (normalized): {retrained_to_approx_normalized}")
        print("--- Angle ---- ")
        #print(f"angle between retrained model and retrained-approx: {round(float(LKO_to_retrained_angle), 4)}")
        print(f"Spearman correlation between retrained model and retrained-approx: {spearman_val}")

    return l2_diffs,  l2_diffs_normalized, update_angle_, model_angle_, spearman_val