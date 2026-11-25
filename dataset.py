"""
Dataset loader and transforms for galaxy morphology classification.
"""

import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CLASSES = ["spiral", "elliptical", "anomaly"]


class Rotate90:
    """Random rotation by 0, 90, 180, or 270 degrees."""
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img):
        return transforms.functional.rotate(img, random.choice(self.angles))


# Backward compatibility alias
RandomAstronomyRotation = Rotate90


def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        Rotate90(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


def get_loaders(data_dir, batch_size=16, img_size=224, num_workers=2):
    train_tf, val_tf = get_transforms(img_size)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Missing train or val folder in {data_dir}")

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_loader = None
    if os.path.exists(test_dir):
        test_set = datasets.ImageFolder(test_dir, transform=val_tf)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_set.classes


# Alias for backward compatibility
create_dataloaders = get_loaders
