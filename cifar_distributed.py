from mmcr.cifar_stl.train import train

import torch
import numpy as np
import random, os
import torch.multiprocessing as mp
import torch.distributed as dist
from argparse import ArgumentParser


os.environ['MASTER_ADDR'] = 'localhost'  
os.environ['MASTER_PORT'] = '12355'

def set_seeds(seed=42):
    """
    Set random seeds to ensure that results can be reproduced.

    Parameters:
        seed (`int`): The random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main_worker(rank, world_size, args):
    set_seeds()
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    args.device = torch.device(f'cuda:{rank}') 

    train(
        dataset=args.dataset,
        n_aug=args.n_aug,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        lmbda=args.lmbda,
        save_folder=args.save_folder,
        save_freq=args.save_freq,
        distributed=True,
        device=args.device,
        world_size=world_size, 
        rank=rank,
    )


def main(args):
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_aug", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.000125)
    parser.add_argument("--lmbda", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=22)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument(
        "--save_folder",
        type=str,
        default="./training_checkpoints_distributed/cifar_stl",
    )

    args = parser.parse_args()

    main(args)
