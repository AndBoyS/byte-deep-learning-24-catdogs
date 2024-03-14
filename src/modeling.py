from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm


class LocalModelType(nn.Module):
    device: torch.device

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass


class LinearCatDogModel(LocalModelType):
    def __init__(
        self,
        image_size: tuple[int, int],
        num_classes: int,
        num_channels: int = 3,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        input_size = image_size[0] * image_size[1] * num_channels

        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),
        )

        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.body(x)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        pred = logits.argmax(dim=1)
        return pred


class Trainer:
    def __init__(
        self,
        model: LocalModelType,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        self.model = model

        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Вместо списков метрик удобнее использовать tensorboard
        self.train_losses: list[float] = []
        self.train_accs: list[float] = []
        self.val_accs: list[float] = []

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)

    def train(self, num_epochs: int) -> None:
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []

        for epoch in tqdm(range(num_epochs), desc="Training:"):
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                self.optimizer.zero_grad()
                loss = self.compute_loss_on_batch(batch)
                loss.backward()
                self.optimizer.step()

                self.train_losses.append(loss.item())
                
            self.model.eval()
            train_acc = evaluate_loader(self.train_loader, self.model)['acc']
            self.train_accs.append(train_acc)

            if self.val_loader is not None:
                val_acc = evaluate_loader(self.val_loader, self.model)['acc']
                self.val_accs.append(val_acc)
                
    def compute_loss_on_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        x: torch.Tensor = batch["image"].float()
        y: torch.Tensor = batch["label_idx"]

        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)

        return self.criterion(logits, y)


def evaluate_loader(loader: DataLoader, model: LocalModelType) -> dict[str, Any]:
    with torch.no_grad():
        model.eval()

        preds_batches = []
        targets_batches = []

        for batch in tqdm(loader, desc="Computing metrics"):
            x: torch.Tensor = batch["image"].float()
            target: torch.Tensor = batch["label_idx"]

            x = x.to(model.device)

            pred = model.predict(x)
            targets_batches.append(target)

            preds_batches.append(pred.cpu())

        targets = torch.concatenate(targets_batches)
        preds = torch.concatenate(preds_batches)

        acc = (preds == targets).float().mean().item()

        return {
            "acc": acc,
        }
