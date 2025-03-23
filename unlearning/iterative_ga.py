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




def iterative_probe_GA(model, probe_model, probe_loader, device, layer_ind, lr = 1e-2, epochs = 8, max_loss = 2.):
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



