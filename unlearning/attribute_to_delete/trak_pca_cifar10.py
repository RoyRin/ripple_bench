from unlearning import training
from unlearning import datasets
import os
import datetime
from pathlib import Path
import wget
# from tqdm.auto import tqdm
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision
import warnings
from trak import TRAKer
from unlearning.datasets.cifar10 import get_cifar_dataset, get_cifar_forget_retain_loaders
from unlearning.models import resnet9
from unlearning.models.resnet9 import ResNet9Mul

from unlearning.eval.nn_evals import evaluate_model, get_losses_and_logits #get_losses_on_dataloader

from unlearning.training.train import train_model_wrapper
warnings.filterwarnings('ignore')



def get_trak_features(model, data_train_loader, ckpts, proj_dim= 4096):
    """

    Args:
        model (_type_): _description_
        data_train_loader (_type_): _description_
        ckpts (_type_): path to the checkpoint files - used for feature extraction
    """

    traker = TRAKer(model=model,
                    task='image_classification',
                    proj_dim=proj_dim,
                    train_set_size=len(data_train_loader.dataset))

    ## Compute TRAK features for train data
    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(ckpt, model_id=model_id)
        ten_percent = len(data_train_loader) // 10
        for b_, batch in enumerate(data_train_loader):
            if b_ % ten_percent == 0:
                print(f"model_id - {model_id} - batch - {b_}/ {len(data_train_loader)}")
            batch = [x.cuda() for x in batch]
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])

        traker.finalize_features()
    print(f"computed TRAK features")
    return traker


def compute_trak_scores(
    traker,
    trak_targets_loader,
    ckpts,
    save_name="quickstart",
):

    for model_id, ckpt in enumerate(ckpts):
        print(f"compute trak scores from chkpt- {model_id}")
        traker.start_scoring_checkpoint(exp_name=save_name,
                                        checkpoint=ckpt,
                                        model_id=model_id,
                                        num_targets=len(
                                            trak_targets_loader.dataset))
        
        for batch in trak_targets_loader:
            batch = [x.cuda() for x in batch]
            traker.score(batch=batch, num_samples=batch[0].shape[0])
    print(f"finalize the scores")
    scores = traker.finalize_scores(exp_name=save_name)
    print(f"returning scores")
    return scores


def compute_gradients(
    traker,
    trak_targets_loader,
    ckpts,
    save_name="quickstart",
    proj_dim = 4096
):
    model_id = len(ckpts) -1
    ckpt = ckpts[model_id]

    print(f"compute gradients from chkpt- {model_id}")
    traker.start_scoring_checkpoint(exp_name=save_name,
                                    checkpoint=ckpt,
                                    model_id=model_id,
                                    num_targets=len(
                                        trak_targets_loader.dataset))
    num_pts = len(trak_targets_loader.dataset)
    #gradients = np.empty((num_pts, proj_dim))
    gradients = []

    for b_, batch in enumerate(trak_targets_loader):
        batch = [x.cuda() for x in batch]
        grads = traker.score(batch=batch, num_samples=batch[0].shape[0]).cpu()
        print(f"{b_}: grads - {type(grads)} - {grads.shape}")
        gradients.append(grads)
    
    return np.vstack(gradients)

    #gradients = np.vstack((gradients, grads))
    


def load_model_from_checkpoints(ckpt, evaluate=False):
    #model = training.construct_rn9().to(
    #    memory_format=torch.channels_last).cuda()
    # HACK - hardcoding the model name now.
    model = ResNet9Mul(num_classes=10, mul=0.14).cuda()

    
    model.load_state_dict(ckpt)
    model = model.eval()

    if evaluate:
        batch_size = 256
        val_dataset = get_cifar_dataset(split='val', augment=True)
        val_dataloader = get_dataloader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

        evaluate_model(model, val_dataloader)
    return model


def get_dataloader(dataset, batch_size=256, num_workers=16, indices=None, shuffle=False,prefetch_factor=2):
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor
    )

def get_singular_vals(gradients, k = 1500, n_iter = 20, centered = True, algo= "arpack"):
    print(f"gradients.shape - {gradients.shape} - {type(gradients)}")
    # gradietns is n x d (for projection dim d (4095))
    # trunscated, ranomdized SVD from sklearn
    from sklearn.decomposition import TruncatedSVD
    # 
    # Compute the mean of each column (feature)
    start_time = datetime.datetime.now()

    print(f"centering")
    if centered:
        column_means = gradients.mean(axis=0)
        # Subtract the mean of each column from the gradients matrix
        centered_gradients = gradients - column_means
    else:
        centered_gradients = gradients 
    frob_norm  = np.linalg.norm(centered_gradients)
    
    print(f"frobenius norm : {frob_norm}")

    print(f"computing SVD:")
    svd = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=0, algorithm =algo)
    svd.fit(centered_gradients)
    s = svd.singular_values_
    # _ , s, _ = np.linalg.svd(gradients)
    return s 


"""
Goal:
    Trak scores should be - for each point f in the forget_set, if you were to remove f, how would it influence point i in some other set.

"""
if __name__ == "__main__":
    do_training = False
    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    #DATA_DIR = Path("/n/home04/rrinberg/data_dir/unlearning")

    DATA_DIR = Path("/n/holyscratch01/vadhan_lab/Lab/rrinberg/unlearning")
    DATA_DIR = Path("/n/home04/rrinberg/data_dir/unlearning-gradients")

    if not DATA_DIR.exists():
        DATA_DIR = Path(os.getcwd())

    SAVE_DIR = DATA_DIR / "trak_unlearning_results"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"data dir- {DATA_DIR}")
    print(f"SAVE_DIR dir- {SAVE_DIR}")
    
    #
    seed = 1
    np.random.seed(seed)
    # fix seed for torch

    forget_set_size = 1000
    batch_size = 512 # 128  # 512
    fulldata_checkpoints_dir = DATA_DIR / "full_checkpoints"
    train_dataset = get_cifar_dataset(split='train', augment=True)
    train_full_dataloader = get_dataloader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    val_dataset = get_cifar_dataset(split='val', augment=True)
    val_dataloader = get_dataloader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    ##### Train full model

    epochs = 100

    ####
    ## Train Models
    ####
    print("training full model")
    checkpoint_epochs = [23,50, 75, 99]
    epochs = 50
    checkpoint_epochs = [23,47,49]
    
    train_model_wrapper(model_suffix="full_train",
                epochs=epochs,
                data_loader=train_full_dataloader,
                checkpoints_dir=fulldata_checkpoints_dir,
                checkpoint_epochs = checkpoint_epochs)


    batch_size = 32
    train_dataset = get_cifar_dataset(split='train', augment=True)
    train_full_dataloader = get_dataloader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    val_dataset = get_cifar_dataset(split='val', augment=True)
    val_dataloader = get_dataloader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    
    ##
    ## acquire list of checkpoints
    ##
    ckpt_files = sorted(list(Path(fulldata_checkpoints_dir).rglob('*.pt')))
    ckpts = [torch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]
    last_ckpt = ckpts[-1]
    #####
    ## extract the original and the retain models
    #####

    # load up the last model

    model = load_model_from_checkpoints(last_ckpt)
    print("retain files!")


    if False:

        ##### Train retain model

        forget_dataloader, retain_dataloader = get_cifar_forget_retain_loaders(
            train_dataset,
            forget_set_size,
            shuffle=True,
            num_workers=8,
            batch_size=batch_size,
            seed=seed)

        print("training retain model")
        retain_checkpoints_dir = DATA_DIR / "retain_checkpoints"

        train_model_wrapper(model_suffix="retain_train",
                    epochs=epochs,
                    data_loader=retain_dataloader,
                    checkpoints_dir=retain_checkpoints_dir)
        retain_ckpt_files = sorted(list(
            Path(retain_checkpoints_dir).rglob('*.pt')))
        retain_last_ckpt = torch.load(retain_ckpt_files[-1], map_location='cpu')
        retain_model = load_model_from_checkpoints(retain_last_ckpt)

        #####
    ## Compute losses on datasets
    #####
    if False:
        print(f"get losses on datasets")
        original_model_on_forget, original_model_on_forget_logits = get_losses_and_logits(
            model, forget_dataloader.dataset)
        
        original_model_on_retain, original_model_on_retain_logits = get_losses_and_logits(
            model, retain_dataloader.dataset)
        
    original_model_on_holdout, original_model_on_holdout_logits = get_losses_and_logits(
        model, val_dataloader)
    original_model_on_train_full, original_model_on_train_full_logits = get_losses_and_logits(
        model, train_full_dataloader)
    if False:
        print(f"retain model")
        retrained_loss_on_forget, retrained_loss_on_forget_logits = get_losses_and_logits(
            retain_model, forget_dataloader.dataset)
        retrained_loss_on_retain, retrained_loss_on_retain_logits = get_losses_and_logits(
            retain_model, retain_dataloader.dataset)
        retrained_loss_on_holdout, retrained_loss_on_holdout_logits = get_losses_and_logits(
            retain_model, val_dataloader.dataset)
        retrained_model_on_train_full, retrained_model_on_train_full_logits = get_losses_and_logits(
            retain_model, train_full_dataloader.dataset)

    losses = {
        # original losses
        #"original_loss_on_forget": original_model_on_forget,
        #"original_loss_on_retain": original_model_on_retain,
        "original_loss_on_holdout": original_model_on_holdout,
        "original_loss_on_full": original_model_on_train_full,
        # retrained losses
        #"retrained_loss_on_forget": retrained_loss_on_forget,
        #"retrained_loss_on_retain": retrained_loss_on_retain,
        #"retrained_loss_on_holdout": retrained_loss_on_holdout,
        #"retrained_loss_on_full": retrained_model_on_train_full,
    }
    for k, v in losses.items():
        save_fn = SAVE_DIR / f"{k}__{date_str}.npy"
        print(f"saving to save_fn")
        print(f"saving losses {k} - {v.shape}")
        np.save(save_fn, v)
    np.savez(SAVE_DIR / f"losses__{date_str}.npz", **losses)
    print(f"completed losses on the different datasets")

    logits = {
        # original losses
        #"original_loss_on_forget": original_model_on_forget_logits,
        #"original_loss_on_retain": original_model_on_retain_logits,
        "original_loss_on_holdout": original_model_on_holdout_logits,
        "original_loss_on_full": original_model_on_train_full_logits,
        # retrained losses
        #"retrained_loss_on_forget": retrained_loss_on_forget_logits,
        #"retrained_loss_on_retain": retrained_loss_on_retain_logits,
        #"retrained_loss_on_holdout": retrained_loss_on_holdout_logits,
        #"retrained_loss_on_full": retrained_model_on_train_full_logits,
    }
    LOGITS_SAVE_DIR = SAVE_DIR / "logits"
    os.makedirs(LOGITS_SAVE_DIR, exist_ok=True)

    for k, v in losses.items():
        save_fn = LOGITS_SAVE_DIR / f"{k}_logits__{date_str}.npy"
        print(f"saving to save_fn")
        print(f"v.shape - {v.shape}")
        np.save(save_fn, v)
    np.savez(LOGITS_SAVE_DIR / f"logits__{date_str}.npz", **logits)
    print(f"completed losses on the different datasets")


    # end goal:
    # get LOO model, Retrained Model, Original model on the retain set
    # get LOO model, Retrained Model, Original model on the forget set
    # get LOO model, Retrained Model, Original model on the validation set

    #####
    ## Compute TRAK scores
    #####

    # get TRAK features for the model
    batch_size = 16
    proj_dim = 4096
    proj_dim = 10_000

    
    print("Create TRAK model")

    #forget_traker = get_trak_features(model,
    #                                  data_train_loader=forget_dataloader,
    #                                  ckpts=ckpts)
    ckpts = [ckpts[-1]]
    print(f"ckpts - {ckpts}")
    train_traker = get_trak_features(model,
                                      data_train_loader=train_full_dataloader,
                                      ckpts=ckpts, proj_dim =proj_dim)

    traker_model = train_traker

    # compute TRAK scores for forget set

    #val_loader = datasets.get_cifar_dataloader(split='val', augment=False)

    grad_save_dir= SAVE_DIR / "gradients"
    os.makedirs(grad_save_dir, exist_ok=True)
    centered = True
    centered_str = "centered" if centered else ""
    k = 2000
    
    ####
    train_save_name = f"trak_scores__full_dataloader__{date_str}"
    print(f"computing gradients! on full set ")
    gradients = compute_gradients(
        traker_model,
        trak_targets_loader=train_full_dataloader,
        save_name=train_save_name, ckpts=ckpts, proj_dim = proj_dim)
    gradients = gradients.astype(np.float32)
    print("FULL SET\n"*3)

    algo = "arpack"
    s = get_singular_vals(gradients, k = k, centered=centered, algo=algo)

    
    grad_save_path = grad_save_dir / f"singular_vals_{algo}__TRAIN__{centered_str}__{proj_dim}__gradients_singular_values__{date_str}.npy"
    np.save(grad_save_path, s)

    grad_save_path = grad_save_dir / f"gradients_TRAIN__{centered_str}__{proj_dim}__gradients_singular_values__{date_str}.npy"
    np.save(grad_save_path, gradients)
        
    print(f"grad_save_path-  {grad_save_path}")
    

    print(f"Compute TRAK on val set")
    train_save_name = f"trak_scores__val_dataloader__{date_str}"

    if False:
        forget_trak_scores = compute_trak_scores(
            traker_model,
            trak_targets_loader=val_dataloader,
            save_name=train_save_name, ckpts=ckpts)
    
    print(f"computing gradients! on val set ")
    gradients = compute_gradients(
        traker_model,
        trak_targets_loader=val_dataloader,
        save_name=train_save_name, ckpts=ckpts, proj_dim = proj_dim)
    gradients = gradients.astype(np.float32)
    
    # compute singular values of the gradients, no need to save vectors, save singular values
    #print(f"gradients.shape - {gradients.shape} - {type(gradients)}")
    #_ , s, _ = np.linalg.svd(gradients)
    print("VAL SET\n"*3)
    
    s = get_singular_vals(gradients, k = k, centered=centered, algo=algo)
    grad_save_path = grad_save_dir / f"singular_vals_{algo}_VALIDATION__{centered_str}__{proj_dim}__gradients_singular_values__{date_str}.npy"

    np.save(grad_save_path, s)

    grad_save_path = grad_save_dir / f"gradients_VALIDATION__{centered_str}__{proj_dim}__gradients_singular_values__{date_str}.npy"
    np.save(grad_save_path, gradients)

    
    print(f"grad_save_path-  {grad_save_path}")
    print(f"computed gradients singular values")
    
    

    print("computed forget scores")

    if False:
        forget_save_name = f"trak_scores__forget_set__{date_str}"
        forget_trak_scores = compute_trak_scores(
            traker_model,
            trak_targets_loader=forget_dataloader,
            save_name=forget_save_name, ckpts=ckpts)

        val_save_name = f"trak_scores__validation_set__{date_str}"
        print(f"Compute TRAK on hold out set (validation data)")
        val_trak_scores = compute_trak_scores(traker_model,
                                            trak_targets_loader=val_dataloader,
                                            save_name=val_save_name, ckpts=ckpts)

        print(f"Compute TRAK on retain set")
        retain_save_name = f"trak_scores__retain_set__{date_str}"
        retain_trak_scores = compute_trak_scores(
            traker_model,
            trak_targets_loader=retain_dataloader,
            save_name=retain_save_name, ckpts=ckpts)