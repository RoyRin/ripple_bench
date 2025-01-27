import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, DistributedSampler

def main(config):
    mp.spawn(train, nprocs=torch.cuda.device_count(), args=(config,))

def train(rank, config):
    dist.init_process_group(backend=config['dist_backend'], init_method=config['dist_url'], world_size=torch.cuda.device_count(), rank=rank)
    torch.cuda.set_device(rank)

    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets and loaders
    train_dataset = datasets.ImageFolder(os.path.join(config['data_path'], 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(config['data_path'], 'val'), transform=transform_val)

    train_sampler = DistributedSampler(train_dataset, num_replicas=torch.cuda.device_count(), rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=config['workers'], pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['workers'], pin_memory=True)

    # Model, optimizer, and loss
    model = models.resnet50()
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    criterion = torch.nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        model.train()

        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.cuda(rank, non_blocking=True), targets.cuda(rank, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0 and i % 10 == 0:
                print(f"Epoch [{epoch+1}/{config['epochs']}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()

        if rank == 0:
            validate(model, val_loader, criterion, rank)

    if rank == 0:
        torch.save(model.state_dict(), "resnet50_imagenet.pth")

def validate(model, val_loader, criterion, rank):
    model.eval()
    total, correct, val_loss = 0, 0, 0

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.cuda(rank, non_blocking=True), targets.cuda(rank, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    config = {
        'data_path': '/path/to/imagenet',
        'batch_size': 256,
        'epochs': 90,
        'lr': 0.1,
        'workers': 8,
        'dist_backend': 'nccl',
        'dist_url': 'env://'
    }
    main(config)
