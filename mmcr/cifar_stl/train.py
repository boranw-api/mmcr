import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import einops
import os

from mmcr.cifar_stl.data import get_datasets
from mmcr.cifar_stl.models import Model
from mmcr.cifar_stl.knn import test_one_epoch
from mmcr.cifar_stl.loss_mmcr import MMCR_Loss
from argparse import ArgumentParser

def train(
    dataset: str,
    n_aug: int,
    batch_size: int,
    lr: float,
    epochs: int,
    lmbda: float,
    save_folder: str,
    save_freq: int,
    distributed: bool,
    device: torch.device,
    world_size: int,  
    rank: int,
):  
    # Dataset
    train_dataset, memory_dataset, test_dataset = get_datasets(
        dataset=dataset, n_aug=n_aug
    )

    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        memory_sampler = torch.utils.data.DistributedSampler(
            memory_dataset, num_replicas=world_size, rank=rank
        )
        test_sampler = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank
        )

        shuffle = False
    else:
        train_sampler = None
        memory_sampler = None
        test_sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=12, sampler=train_sampler
    )
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=128, shuffle=False, num_workers=12, sampler=memory_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=12, sampler=test_sampler
    )

    # Model and optimizer
    model = Model(projector_dims=[512, 128], dataset=dataset)
    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_function = MMCR_Loss(lmbda=lmbda, n_aug=n_aug, distributed=distributed)

    top_acc = 0.0
    for epoch in range(epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss, total_num = 0.0, 0

        # Progress bar for rank 0 only
        if rank == 0:
            train_bar = tqdm(train_loader)
        else:
            train_bar = train_loader

        for step, data_tuple in enumerate(train_bar):
            optimizer.zero_grad()

            # Forward pass
            img_batch, labels = data_tuple
            img_batch = einops.rearrange(img_batch, "B N C H W -> (B N) C H W")
            img_batch = img_batch.to(device, non_blocking=True)
            features, out = model(img_batch)
            loss, loss_dict = loss_function(out)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the training bar
            total_num += data_tuple[0].size(0)
            total_loss += loss.item() * data_tuple[0].size(0)
            if rank == 0:
                train_bar.set_description(
                    "Train Epoch: [{}/{}] Loss: {:.4f}".format(
                        epoch+1, epochs, total_loss / total_num
                    )
                )

        if epoch % 1 == 0:
            acc_1, acc_5 = test_one_epoch(
                model,
                memory_loader,
                test_loader,
            )

            if rank == 0:
                if acc_1 > top_acc:
                    top_acc = acc_1

                if epoch % save_freq == 0 or acc_1 == top_acc:
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    torch.save(
                        model.state_dict(),
                        f"{save_folder}/{dataset}_{n_aug}_{epoch}_acc_{acc_1:0.2f}.pth",
                    )
