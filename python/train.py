#!/usr/bin/env python3
"""Script for CNN training.

"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from ml import load_data, get_outputs_processor, CNN, data_transform
from utils import set_seed, split_indices, print_model, Trainer


def main() -> None:
    # Setup
    seed = 3019
    set_seed(seed=seed)

    # Data preprocessing
    inputs, outputs = load_data()

    train_indices, valid_indices, _ = split_indices(num_samples=inputs.shape[0],
                                                    percent_train=0.9,
                                                    percent_valid=0.05)

    inputs_train = inputs[train_indices]
    outputs_train = outputs[train_indices]
    inputs_valid = inputs[valid_indices]
    outputs_valid = outputs[valid_indices]

    outputs_processor = get_outputs_processor()

    outputs_train = outputs_processor.fit_transform(outputs_train)
    outputs_valid = outputs_processor.transform(outputs_valid)

    # Datasets
    train_dataset = TensorDataset(torch.from_numpy(inputs_train).to(dtype=torch.float32),
                                  torch.from_numpy(outputs_train).to(dtype=torch.float32))

    valid_dataset = TensorDataset(torch.from_numpy(inputs_valid).to(dtype=torch.float32),
                                  torch.from_numpy(outputs_valid).to(dtype=torch.float32))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=64,
                              shuffle=True)

    # Model
    model = CNN()

    print_model(model)

    # Criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)

    # Train
    model_dir = Path(__file__).resolve().parent.parent / 'models'

    trainer = Trainer(model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      valid_loader=valid_loader,
                      epochs=1000,
                      resume=True,
                      save_file=model_dir / 'model.pt',
                      data_transform=data_transform)

    trainer.train(verbose=True)


if __name__ == '__main__':
    main()
