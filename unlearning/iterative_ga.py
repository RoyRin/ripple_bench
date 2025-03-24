import torch 
import torch.nn as nn
import torch.nn.functional as F


def get_flattened_embedding__model(model: nn.Module, x: torch.Tensor, layer_ind: int):
    embeddings = []
    model_device = next(model.parameters()).device
    x = x.to(model_device)

    def hook(module, input, output):
        embeddings.append(output)

    layers = list(model.children())

    if layer_ind < 0 or layer_ind >= len(layers):
        raise ValueError(f"layer_num must be between 0 and {len(layers)-1}")

    handle = layers[layer_ind].register_forward_hook(hook)

    # Forward pass (without torch.no_grad()!)
    model(x)

    handle.remove()

    embedding = embeddings[0].flatten(start_dim=1)  # ← NO detach()

    return embedding



def freeze_all_but_layer_by_index(model, unfreeze_layer_inds):
    """
    Freeze all layers except the one at the given index from model.children().

    Args:
        model: the ResNet model.
        layer_ind: integer index from model.children() specifying the layer to unfreeze.
                   (Typically, 4:layer1, 5:layer2, 6:layer3, 7:layer4 in ResNet)
    """

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False


    for layer_ind, layer in enumerate(model.named_children()):

        if layer_ind in unfreeze_layer_inds:
            print(f"unfreezing layer {layer_ind}")
            for param in layer[1].parameters():
                param.requires_grad = True




def iterative_probe_GA(model, probe_model, probe_loader, device, layer_ind, lr = 1e-2, epochs = 8, max_loss = 2., **kwargs):
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #SGD for GA
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    criterion = nn.MSELoss()

    # train model, freeze probe
    for param in model.parameters():
        param.requires_grad = True

    for param in probe_model.parameters():
        param.requires_grad = False


    freeze_layers= [layer_ind-1, layer_ind]
    freeze_all_but_layer_by_index(model, freeze_layers)

    # count unfrozen params
    unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen parameters count: {unfrozen_params}")

    model.train()
    probe_model.eval()  # Probe is frozen (eval mode recommended)

    report_every = (len(probe_loader) // 5) +1

    # Let's try doing Gradient Ascent on the model parameters

    break_out = False
    for epoch in range(epochs):  # Train for a few epochs
        for probe_ind, (images, labels) in enumerate(probe_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass through model to get embeddings
            embeddings = get_flattened_embedding__model(model, images, layer_ind=layer_ind)

            # Forward pass through probe (frozen)
            predictions = probe_model(embeddings)
            # squeeze predictions
            predictions = predictions.squeeze()

            # predictions 
            # Compute loss
            loss = criterion(predictions, labels)
            ascent_loss = -1 * loss
            ascent_loss.backward()
            # Backward pass: updates model only, probe is frozen
            #loss.backward()
            optimizer.step()
            if probe_ind % report_every == 0:
                print(f"Epoch [{epoch + 1}], Step [{probe_ind + 1}/{len(probe_loader)}], Loss: {loss.item():.4f}")
            if loss > max_loss:
                break_out= True
                break
        print(f"{epoch} -- Loss: {loss.item():.4f}")
        if break_out:
            break

    return model





def iterative_probe_GA_maximize_KL(model, probe_model, probe_loader, device,  layer_ind, loss_fn_type= "mse", lr = 1e-2, epochs = 8, min_loss = .01):
    """
    optimziation won't be maximizing MSE with the true labels, but will instead maximize KL divergence instead - this only works given that the probe dataset is split 50-50
    """
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    

    # train model, freeze probe
    for param in model.parameters():
        param.requires_grad = True

    for param in probe_model.parameters():
        param.requires_grad = False


    freeze_layers= [layer_ind-1, layer_ind]
    freeze_all_but_layer_by_index(model, freeze_layers)

    # count unfrozen params
    unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen parameters count: {unfrozen_params}")

    model.train()
    probe_model.eval()  # Probe is frozen (eval mode recommended)

    report_every = (len(probe_loader) // 5) +1

    # Let's try doing Gradient Ascent on the model parameters
    ###
    #
    ###


    if loss_fn_type == 'mse':
        # For MSE loss, the target probability is 0.5 (output shape: [batch_size, 1])
        artificial_y_target = torch.full((100, 1), 0.5).to(device)
        
        loss_fn = nn.MSELoss()
        
    elif loss_fn_type == 'kl':
        # For KL divergence, the target is the uniform distribution: [0.5, 0.5] (output shape: [batch_size, 2])
        #artificial_y_target = torch.full((100, 2), 0.5).to(device)
        artificial_y_target = torch.full((100, 1), 0.5).to(device)
        
        loss_fn = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError("loss_fn_type must be either 'mse' or 'kl'.")
    

    break_out = False
    for epoch in range(epochs):  # Train for a few epochs
        for probe_ind, (images, labels) in enumerate(probe_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass through model to get embeddings
            embeddings = get_flattened_embedding__model(model, images, layer_ind=layer_ind)

            # Forward pass through probe (frozen)
            predictions = probe_model(embeddings) # [batch_size, 2]
            # squeeze predictions
            predictions = predictions.squeeze()
            #print(f"predictions shape: {predictions.shape}")
            #print(f"predictions: {predictions}")

            
            probabilities = torch.stack([1 - predictions, predictions], dim=1)
            #print(probabilities.shape)  # should be [64, 2]
            log_probabilities = torch.log(probabilities)
            
        

            # predictions 
            batch_size = images.size(0)
            # Compute loss
            if loss_fn_type == 'mse':
                artificial_y_target = torch.full((batch_size, 1), 0.5).to(device)
                loss = loss_fn(predictions, artificial_y_target)
            else:
                artificial_y_target = torch.full((batch_size, 2), 0.5).to(device)
                if False:
                    # slightly skew the target in the wrong direction
                    artificial_y_target = torch.stack([1-labels, labels], dim=1).to(device)
                    wrong_y_target = 1 - artificial_y_target
                    
                    #artificial_y_target = wrong_y_target

                    artificial_y_target = 1- torch.abs(.3 - artificial_y_target)
                
                loss = loss_fn(log_probabilities, artificial_y_target)
                #print(f"Loss: {loss.item()}")
                #print(f"log_probabilities: {log_probabilities}")
            #loss = criterion(predictions, labels)
            loss.backward()
            # Backward pass: updates model only, probe is frozen
            #loss.backward()
            optimizer.step()
            if probe_ind % report_every == 0:
                print(f"Epoch [{epoch + 1}], Step [{probe_ind + 1}/{len(probe_loader)}], Loss: {loss.item():.4f}")
            if loss < min_loss:
                break_out= True
                break
        print(f"{epoch} -- Loss: {loss.item():.4f}")
        if break_out:
            break

    return model






