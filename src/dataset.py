from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CatDogDataset(Dataset):
    def __init__(self, base_folder: Path, transform: Callable | None = None) -> None:
        self.image_paths = sorted(base_folder.rglob("*.jpg"))
        
        self.transform = transform
        self.labels = list(set(path.parent.name for path in self.image_paths))
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        
        label = image_path.parent.name
        label_idx = self.label_to_idx[label]

        return {
            "image": image,
            "label_idx": label_idx,
        }
