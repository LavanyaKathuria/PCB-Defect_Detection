"""
dataset_loader.py

Dataloader used for DeepPCB patch-based training.
Loads image patches using bounding box annotations.
"""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class DeepPCBPatchDataset(Dataset):
    # Dataset for loading cropped PCB defect patches
    def __init__(self, root_dir, split_file, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # split file contains image path and annotation path per line
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue

                img_rel, ann_rel = parts
                img_path = self.root_dir / img_rel
                ann_path = self.root_dir / ann_rel

                # some images were renamed during preprocessing
                if not img_path.exists():
                    alt_img_path = img_path.with_name(
                        img_path.stem + "_temp" + img_path.suffix
                    )
                    if alt_img_path.exists():
                        img_path = alt_img_path
                    else:
                        continue

                if not ann_path.exists():
                    continue

                # read bounding boxes from annotation file
                with open(ann_path, "r") as af:
                    for line in af:
                        x_min, y_min, x_max, y_max, cls_id = map(
                            int, line.strip().split()
                        )

                        # store one entry per bounding box
                        self.samples.append(
                            (img_path, (x_min, y_min, x_max, y_max), cls_id - 1)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, cls_id = self.samples[idx]

        # load full image
        img = Image.open(img_path).convert("RGB")

        # crop defect region
        x_min, y_min, x_max, y_max = bbox
        patch = img.crop((x_min, y_min, x_max, y_max))

        if self.transform:
            patch = self.transform(patch)

        return patch.float(), torch.tensor(cls_id, dtype=torch.long)


def get_dataloaders(
    root_dir,
    train_split,
    test_split,
    image_size=224,
    batch_size=32,
    num_workers=0,
    pin_memory=False,
    augment=False
):
    # data augmentation only used during training
    if augment:
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    train_dataset = DeepPCBPatchDataset(
        root_dir, train_split, transform=transform
    )

    # using test split for val here
    val_dataset = DeepPCBPatchDataset(
        root_dir, test_split, transform=transform
    )

    test_dataset = DeepPCBPatchDataset(
        root_dir, test_split, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
