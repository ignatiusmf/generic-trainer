import torchvision.transforms as transforms
import torchvision
import torch
import os
from torchvision.datasets import ImageFolder


def get_loaders(dataset: str, batch_size: int, data_root: str = "./data"):
    """Return train & test loaders for CIFAR-10/100 or Tiny‑ImageNet‑200.

    Args:
        dataset: "Cifar10", "Cifar100" or "TinyImageNet" (case‑sensitive)
        batch_size: batch size for the dataloaders
        data_root: root directory where data lives / will be downloaded
    """
    # ── dataset‑specific metadata ──────────────────────────────────────────────
    if dataset == "Cifar100":
        ds = torchvision.datasets.CIFAR100
        mean, std = (0.5071, 0.4867, 0.4409), (0.267, 0.256, 0.276)
        input_size = 32
    elif dataset == "Cifar10":
        ds = torchvision.datasets.CIFAR10
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        input_size = 32
    elif dataset == "TinyImageNet":
        # Tiny‑ImageNet‑200 isn’t bundled with torchvision, so we’ll load it via
        # ImageFolder once it has been downloaded & extracted:
        #   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
        #   unzip tiny-imagenet-200.zip -d ./data
        # Directory structure after extraction:
        #   ./data/tiny-imagenet-200/train/*
        #   ./data/tiny-imagenet-200/val/images/*
        ds = None  # handled separately below
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        input_size = 64
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    num_workers = 8 if os.name == "posix" else 0

    # ── transforms ────────────────────────────────────────────────────────────
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(input_size + 8),  # e.g. 72 → 64 center‑crop
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── dataset objects ───────────────────────────────────────────────────────
    if dataset in {"Cifar10", "Cifar100"}:
        trainset = ds(root=data_root, train=True, download=True, transform=transform_train)
        testset = ds(root=data_root, train=False, download=True, transform=transform_test)
    else:  # Tiny‑ImageNet via ImageFolder
        train_dir = os.path.join(data_root, "tiny-imagenet-200", "train")
        val_dir   = os.path.join(data_root, "tiny-imagenet-200", "val", "images")
        trainset = ImageFolder(train_dir, transform=transform_train)
        testset  = ImageFolder(val_dir,   transform=transform_test)

    # ── dataloaders ───────────────────────────────────────────────────────────
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return trainloader, testloader


class DataHelper:
    """Simple wrapper exposing `.trainloader`, `.testloader`, and metadata."""

    def __init__(self, dataset: str, batch_size: int, data_root: str = "./data"):
        self.trainloader, self.testloader = get_loaders(dataset, batch_size, data_root)
        self.name = dataset
        if dataset == "Cifar100":
            self.class_num = 100
        elif dataset == "Cifar10":
            self.class_num = 10
        elif dataset == "TinyImageNet":
            self.class_num = 200
        else:
            raise ValueError(f"Unknown dataset: {dataset}")


# ── Convenience constructors ────────────────────────────────────────────────

def Cifar10(batch_size: int = 128, data_root: str = "./data") -> DataHelper:
    return DataHelper("Cifar10", batch_size, data_root)


def Cifar100(batch_size: int = 128, data_root: str = "./data") -> DataHelper:
    return DataHelper("Cifar100", batch_size, data_root)


def TinyImageNet(batch_size: int = 128, data_root: str = "./data") -> DataHelper:
    return DataHelper("TinyImageNet", batch_size, data_root)
